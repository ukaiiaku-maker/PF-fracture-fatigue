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

EXPECTED_BRANCH=${EXPECTED_BRANCH:-v10.2.30-hazard-energy-gated-fatigue-events}
CURRENT_BRANCH=$(git branch --show-current)
if [[ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" ]]; then
  echo "ERROR: expected branch=$EXPECTED_BRANCH; observed $CURRENT_BRANCH" >&2
  exit 2
fi
EXPECTED_HEAD=${EXPECTED_HEAD:-}
ACTUAL_HEAD=$(git rev-parse HEAD)
if [[ -n "$EXPECTED_HEAD" && "$ACTUAL_HEAD" != "$EXPECTED_HEAD" ]]; then
  echo "ERROR: expected HEAD=$EXPECTED_HEAD; observed $ACTUAL_HEAD" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: working tree is not clean" >&2
  git status --short >&2
  exit 2
fi

DELTA_K_MPA_SQRT_M=${DELTA_K_MPA_SQRT_M:-6.9866145600638339}
HAZARD_SEED=${HAZARD_SEED:-2001726}
PARAMETER_OPTION=${PARAMETER_OPTION:-v913_paper_weakT01_0129902_persistent_sites}
FAMILY_JSON=${FAMILY_JSON:-$ROOT/runs/v10_2_28_kernel_cache/4fa015d77f1aadf05f77f550366f64cd611f537ae716bbd47870bf9e6fe2f873/family.json}
RUN_CYCLES_MAX=${RUN_CYCLES_MAX:-1e6}
REFERENCE_HORIZON_CYCLES=${REFERENCE_HORIZON_CYCLES:-1e10}
OUTER_PROPOSAL_CYCLES=${OUTER_PROPOSAL_CYCLES:-1e6}
STEPS=${STEPS:-16}
MAX_WALL_SECONDS=${MAX_WALL_SECONDS:-900}
OUTROOT=${OUTROOT:-$ROOT/runs/v10_2_30_weakt_0p55_transient_$(date +%Y%m%d_%H%M%S)}

[[ -s "$FAMILY_JSON" ]] || {
  echo "ERROR: missing FAMILY_JSON=$FAMILY_JSON" >&2
  exit 2
}
if [[ -e "$OUTROOT" ]]; then
  echo "ERROR: output already exists: $OUTROOT" >&2
  exit 2
fi
mkdir -p "$OUTROOT"

unset V10230_ACTIVE_STATE_BLOCK_CONTROL || true
unset V10230_FEEDBACK_STATE_BLOCK_CONTROL || true
unset V10230_VHCF_RELATIVE_CYCLE_TOL || true
unset V10230_VHCF_GROWTH_FACTOR || true
unset V10229_COUPLED_HAZARD_ABS_DB_TOL || true
unset V10229_COUPLED_HAZARD_STATE_TARGET_FRACTION || true

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PARAMETER_CAMPAIGN=1
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

export V10230_FORWARD_OUTER_PROPOSAL_CYCLES="$OUTER_PROPOSAL_CYCLES"
export V10230_FORWARD_INITIAL_CYCLES=${V10230_FORWARD_INITIAL_CYCLES:-1e-3}
export V10230_FORWARD_GROWTH_FACTOR=${V10230_FORWARD_GROWTH_FACTOR:-2}
export V10230_FORWARD_SHRINK_FACTOR=${V10230_FORWARD_SHRINK_FACTOR:-0.5}
export V10230_FORWARD_CLOCK_REL_TOL=${V10230_FORWARD_CLOCK_REL_TOL:-1e-3}
export V10230_FORWARD_SHIELD_REL_TOL=${V10230_FORWARD_SHIELD_REL_TOL:-1e-3}
export V10230_FORWARD_SIGMA_REL_TOL=${V10230_FORWARD_SIGMA_REL_TOL:-1e-3}
export V10230_FORWARD_RADIUS_REL_TOL=${V10230_FORWARD_RADIUS_REL_TOL:-1e-3}
export V10230_FORWARD_LOG_LAMBDA_TOL_DECADES=${V10230_FORWARD_LOG_LAMBDA_TOL_DECADES:-0.01}
export V10230_FORWARD_EVENT_LOCALIZATION_CYCLES=${V10230_FORWARD_EVENT_LOCALIZATION_CYCLES:-1e-6}
export V10230_FORWARD_MAX_ACCEPTED_SEGMENTS=${V10230_FORWARD_MAX_ACCEPTED_SEGMENTS:-32}
export V10230_FORWARD_MAX_TRIAL_INTEGRATIONS=${V10230_FORWARD_MAX_TRIAL_INTEGRATIONS:-128}
export V10230_FORWARD_HEARTBEAT_SEGMENTS=${V10230_FORWARD_HEARTBEAT_SEGMENTS:-4}

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import arrhenius_fracture
from arrhenius_fracture import persistent_site_forward_coupled_hazard_v10230 as forward
from arrhenius_fracture import persistent_site_forward_selector_v10230 as selector

print("v10.2.30 weak-T transient diagnostic")
print(f"  package={Path(arrhenius_fracture.__file__).resolve()}")
print(f"  marcher={forward.MODEL_ID}")
print(f"  selector={selector.MODEL_ID}")
print("  stationary_tail_propagation=off")
print("  full_four_class_campaign=off")
PY

"$PYTHON_BIN" - "$OUTROOT" "$ACTUAL_HEAD" "$RUN_CYCLES_MAX" \
  "$REFERENCE_HORIZON_CYCLES" "$OUTER_PROPOSAL_CYCLES" "$STEPS" \
  "$MAX_WALL_SECONDS" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
payload = {
    "schema": "v10.2.30_weakt_transient_run_manifest_v1",
    "git_head": sys.argv[2],
    "run_cycles_max": float(sys.argv[3]),
    "reference_horizon_cycles": float(sys.argv[4]),
    "outer_proposal_cycles": float(sys.argv[5]),
    "steps": int(sys.argv[6]),
    "maximum_wall_seconds": int(sys.argv[7]),
    "stationary_tail_propagation_enabled": False,
    "production_campaign": False,
}
(root / "transient_diagnostic_manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY

LOG="$OUTROOT/run.log"
TIMEOUT_MARKER="$OUTROOT/watchdog_timeout.txt"
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
  --steps "$STEPS" --cycles-max "$RUN_CYCLES_MAX" \
  --block-cycles "$RUN_CYCLES_MAX" --max-block-cycles "$RUN_CYCLES_MAX" \
  --out "$OUTROOT" \
  > >(tee "$LOG") 2>&1 &
PID=$!

(
  sleep "$MAX_WALL_SECONDS"
  if kill -0 "$PID" 2>/dev/null; then
    printf 'watchdog timeout after %s seconds\n' "$MAX_WALL_SECONDS" \
      | tee "$TIMEOUT_MARKER" >&2
    if command -v sample >/dev/null 2>&1; then
      sample "$PID" 5 -file "$OUTROOT/process_sample.txt" >/dev/null 2>&1 || true
    fi
    kill -TERM "$PID" 2>/dev/null || true
  fi
) &
WATCHDOG_PID=$!

wait "$PID"
RC=$?
kill "$WATCHDOG_PID" 2>/dev/null || true
wait "$WATCHDOG_PID" 2>/dev/null || true
set -e

END=$(date +%s)
if [[ -f "$TIMEOUT_MARKER" ]]; then
  RC=124
fi
printf '%s\n' "$RC" > "$OUTROOT/exit_code.txt"
printf '%s\n' "$((END - START))" > "$OUTROOT/wall_seconds.txt"

if [[ -s "$OUTROOT/kinetic_tip_cell_audit_v101.json" ]]; then
  "$PYTHON_BIN" scripts/analyze_v10_2_30_weakt_transient_diagnostic.py \
    "$OUTROOT" \
    --horizon-cycles "$REFERENCE_HORIZON_CYCLES" \
    | tee "$OUTROOT/transient_analysis_console.log"
else
  echo "WARNING: no kinetic audit was written" >&2
fi

echo "exit_code=$RC"
echo "wall_seconds=$((END - START))"
echo "output=$OUTROOT"
echo "run_log=$LOG"
exit "$RC"
