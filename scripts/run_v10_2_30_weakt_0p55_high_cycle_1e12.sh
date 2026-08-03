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
EXPECTED_HEAD=${EXPECTED_HEAD:-}
CURRENT_BRANCH=$(git branch --show-current)
ACTUAL_HEAD=$(git rev-parse HEAD)
if [[ "$CURRENT_BRANCH" != "$EXPECTED_BRANCH" ]]; then
  echo "ERROR: expected branch=$EXPECTED_BRANCH; observed $CURRENT_BRANCH" >&2
  exit 2
fi
if [[ -n "$EXPECTED_HEAD" && "$ACTUAL_HEAD" != "$EXPECTED_HEAD" ]]; then
  echo "ERROR: expected HEAD=$EXPECTED_HEAD; observed $ACTUAL_HEAD" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: working tree is not clean" >&2
  git status --short >&2
  exit 2
fi

FAMILY_JSON=${FAMILY_JSON:-$ROOT/runs/v10_2_28_kernel_cache/4fa015d77f1aadf05f77f550366f64cd611f537ae716bbd47870bf9e6fe2f873/family.json}
PARAMETER_OPTION=${PARAMETER_OPTION:-v913_paper_weakT01_0129902_persistent_sites}
DELTA_K_MPA_SQRT_M=${DELTA_K_MPA_SQRT_M:-6.9866145600638339}
HAZARD_SEED=${HAZARD_SEED:-2001726}
CYCLES_MAX=${CYCLES_MAX:-1e12}
TARGET_EXT_UM=${TARGET_EXT_UM:-25}
STEPS=${STEPS:-20000}
MAX_WALL_SECONDS=${MAX_WALL_SECONDS:-7200}
OUTROOT=${OUTROOT:-$ROOT/runs/v10_2_30_weakt_0p55_high_cycle_1e12_$(date +%Y%m%d_%H%M%S)}

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

# The outer driver supplies the full requested physical horizon. The production
# high-cycle state machine is authoritative and chooses transient, periodic,
# stationary-tail, projective, and event-guard operations internally.
export V10230_FORWARD_OUTER_PROPOSAL_CYCLES="$CYCLES_MAX"

# Reference transient kernel. It is used only when the high-cycle engine rejects
# a periodic/projective operation or needs exact event localization.
export V10230_FORWARD_INITIAL_CYCLES=${V10230_FORWARD_INITIAL_CYCLES:-1e-3}
export V10230_FORWARD_GROWTH_FACTOR=${V10230_FORWARD_GROWTH_FACTOR:-2}
export V10230_FORWARD_SHRINK_FACTOR=${V10230_FORWARD_SHRINK_FACTOR:-0.5}
export V10230_FORWARD_CLOCK_REL_TOL=${V10230_FORWARD_CLOCK_REL_TOL:-1e-3}
export V10230_FORWARD_SHIELD_REL_TOL=${V10230_FORWARD_SHIELD_REL_TOL:-1e-3}
export V10230_FORWARD_SIGMA_REL_TOL=${V10230_FORWARD_SIGMA_REL_TOL:-1e-3}
export V10230_FORWARD_RADIUS_REL_TOL=${V10230_FORWARD_RADIUS_REL_TOL:-1e-3}
export V10230_FORWARD_LOG_LAMBDA_TOL_DECADES=${V10230_FORWARD_LOG_LAMBDA_TOL_DECADES:-0.01}
export V10230_FORWARD_STATE_PROFILE_REL_TOL=${V10230_FORWARD_STATE_PROFILE_REL_TOL:-1e-4}
export V10230_FORWARD_MOBILE_REL_TOL=${V10230_FORWARD_MOBILE_REL_TOL:-1e-4}
export V10230_FORWARD_RETAINED_REL_TOL=${V10230_FORWARD_RETAINED_REL_TOL:-1e-4}
export V10230_FORWARD_BACKSTRESS_REL_TOL=${V10230_FORWARD_BACKSTRESS_REL_TOL:-1e-4}
export V10230_FORWARD_EMISSION_LOG_RATE_TOL_DECADES=${V10230_FORWARD_EMISSION_LOG_RATE_TOL_DECADES:-0.01}
export V10230_FORWARD_MAX_SEGMENT_CYCLES=${V10230_FORWARD_MAX_SEGMENT_CYCLES:-1e6}
export V10230_FORWARD_EVENT_LOCALIZATION_CYCLES=${V10230_FORWARD_EVENT_LOCALIZATION_CYCLES:-1e-6}
export V10230_FORWARD_MAX_ACCEPTED_SEGMENTS=${V10230_FORWARD_MAX_ACCEPTED_SEGMENTS:-256}
export V10230_FORWARD_MAX_TRIAL_INTEGRATIONS=${V10230_FORWARD_MAX_TRIAL_INTEGRATIONS:-1024}
export V10230_FORWARD_HEARTBEAT_SEGMENTS=${V10230_FORWARD_HEARTBEAT_SEGMENTS:-16}

# Periodic-state and stationary-tail admission.
export V10230_PERIODIC_RELATIVE_TOL=${V10230_PERIODIC_RELATIVE_TOL:-1e-8}
export V10230_PERIODIC_DIAGNOSTIC_TOL=${V10230_PERIODIC_DIAGNOSTIC_TOL:-1e-6}
export V10230_PERIODIC_MAX_ITERATIONS=${V10230_PERIODIC_MAX_ITERATIONS:-40}
export V10230_PERIODIC_ANDERSON_DEPTH=${V10230_PERIODIC_ANDERSON_DEPTH:-6}
export V10230_PERIODIC_DAMPING=${V10230_PERIODIC_DAMPING:-0.8}
export V10230_HIGH_CYCLE_STATIONARY_REL_TOL=${V10230_HIGH_CYCLE_STATIONARY_REL_TOL:-1e-7}
export V10230_HIGH_CYCLE_STATIONARY_DIAGNOSTIC_TOL=${V10230_HIGH_CYCLE_STATIONARY_DIAGNOSTIC_TOL:-1e-5}
export V10230_HIGH_CYCLE_STATIONARY_ADMISSION_DISTANCE=${V10230_HIGH_CYCLE_STATIONARY_ADMISSION_DISTANCE:-1e-6}
export V10230_HIGH_CYCLE_EVENT_GUARD_CYCLES=${V10230_HIGH_CYCLE_EVENT_GUARD_CYCLES:-2}

# Slow-manifold propagation is accepted only after an exact-cycle burst and an
# independent map evaluation at the projected endpoint.
export V10230_PROJECTIVE_BURST_CYCLES=${V10230_PROJECTIVE_BURST_CYCLES:-8}
export V10230_PROJECTIVE_MIN_CYCLES=${V10230_PROJECTIVE_MIN_CYCLES:-16}
export V10230_PROJECTIVE_INITIAL_FACTOR=${V10230_PROJECTIVE_INITIAL_FACTOR:-16}
export V10230_PROJECTIVE_GROWTH_FACTOR=${V10230_PROJECTIVE_GROWTH_FACTOR:-100}
export V10230_PROJECTIVE_MAX_CYCLES=${V10230_PROJECTIVE_MAX_CYCLES:-1e12}
export V10230_PROJECTIVE_DRIFT_REL_TOL=${V10230_PROJECTIVE_DRIFT_REL_TOL:-1e-4}
export V10230_PROJECTIVE_HAZARD_REL_TOL=${V10230_PROJECTIVE_HAZARD_REL_TOL:-1e-4}
export V10230_PROJECTIVE_CURVATURE_REL_TOL=${V10230_PROJECTIVE_CURVATURE_REL_TOL:-0.02}
export V10230_PROJECTIVE_MAX_ATTEMPTS=${V10230_PROJECTIVE_MAX_ATTEMPTS:-12}

export V10230_HIGH_CYCLE_TRANSIENT_FALLBACK_CYCLES=${V10230_HIGH_CYCLE_TRANSIENT_FALLBACK_CYCLES:-64}
export V10230_HIGH_CYCLE_MAX_MODE_OPERATIONS=${V10230_HIGH_CYCLE_MAX_MODE_OPERATIONS:-256}
export V10230_HIGH_CYCLE_HEARTBEAT_OPERATIONS=${V10230_HIGH_CYCLE_HEARTBEAT_OPERATIONS:-2}

"$PYTHON_BIN" - "$OUTROOT" "$ACTUAL_HEAD" "$CYCLES_MAX" <<'PY'
import json
import os
import sys
from pathlib import Path

from arrhenius_fracture import persistent_site_high_cycle_engine_v10230 as high
from arrhenius_fracture import persistent_site_high_cycle_state_v10230 as state
from arrhenius_fracture import persistent_site_poincare_v10230 as poincare
from arrhenius_fracture import persistent_site_periodic_solver_v10230 as periodic
from arrhenius_fracture import persistent_site_high_cycle_propagation_v10230 as propagation

root = Path(sys.argv[1])
payload = {
    "schema": "v10.2.30_weakt_0p55_native_high_cycle_1e12_v1",
    "git_head": sys.argv[2],
    "cycles_max": float(sys.argv[3]),
    "production_high_cycle_engine": high.MODEL_ID,
    "active_state_model": state.MODEL_ID,
    "poincare_model": poincare.MODEL_ID,
    "periodic_solver_model": periodic.MODEL_ID,
    "high_cycle_propagation_model": propagation.MODEL_ID,
    "empirical_fatigue_growth_law": False,
    "stationary_tail_draws_rng": False,
    "full_four_class_campaign": False,
    "environment": {
        key: os.environ[key]
        for key in sorted(os.environ)
        if key.startswith("V10230_")
    },
}
(root / "high_cycle_run_manifest.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
print("v10.2.30 real weak-T 0.55 high-cycle production run")
print(f"  high_cycle_engine={high.MODEL_ID}")
print(f"  cycles_max={float(sys.argv[3]):.9g}")
print("  stationary_tail=exact_existing_threshold")
print("  slow_manifold=validated_projective")
print("  post_event_restart=on")
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
  --da-phys 5e-6 --target-crack-extension-um "$TARGET_EXT_UM" \
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
  --steps "$STEPS" --cycles-max "$CYCLES_MAX" \
  --block-cycles "$CYCLES_MAX" --max-block-cycles "$CYCLES_MAX" \
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
  "$PYTHON_BIN" - "$OUTROOT" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
data = json.loads((root / "kinetic_tip_cell_audit_v101.json").read_text())
records = data.get("records", data if isinstance(data, list) else [])
mode_counts = Counter()
cycles = 0.0
fired = 0
partial = 0
for row in records:
    cycles += float(row.get("cycles_consumed", 0.0))
    fired += int(bool(row.get("fired", False)))
    partial += int(bool(row.get("coupled_hazard_partial_return", False)))
    for mode in row.get("coupled_hazard_modes", []):
        mode_counts[str(mode.get("mode", "unknown"))] += 1
summary = {
    "schema": "v10.2.30_weakt_0p55_high_cycle_summary_v1",
    "record_count": len(records),
    "cycles_consumed": cycles,
    "fired_records": fired,
    "partial_returns": partial,
    "mode_counts": dict(sorted(mode_counts.items())),
    "high_cycle_engine_records": sum(
        int(bool(row.get("coupled_hazard_high_cycle_engine", False)))
        for row in records
    ),
    "reached_1e12_or_event": bool(cycles >= 1.0e12 or fired > 0),
}
(root / "high_cycle_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(summary, indent=2, sort_keys=True))
if records and summary["high_cycle_engine_records"] != len(records):
    raise SystemExit("not every cyclic record used the high-cycle engine")
PY
else
  echo "WARNING: no kinetic audit was written" >&2
fi

echo "exit_code=$RC"
echo "wall_seconds=$((END - START))"
echo "output=$OUTROOT"
echo "run_log=$LOG"
exit "$RC"
