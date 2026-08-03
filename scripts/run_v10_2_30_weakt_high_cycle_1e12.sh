#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
TARGET_DELTAK=${TARGET_DELTAK:-${DELTA_K_MPA_SQRT_M:-}}
TARGET_FRACTION=${TARGET_FRACTION:-custom}
RUN_LABEL=${RUN_LABEL:-weakt_${TARGET_FRACTION}}
TARGET_EXT_UM=${TARGET_EXT_UM:-100}
CYCLES_MAX=${CYCLES_MAX:-1e12}
HAZARD_SEED=${HAZARD_SEED:-2001726}

if [[ -z "$TARGET_DELTAK" ]]; then
  echo "ERROR: set TARGET_DELTAK in MPa*sqrt(m)" >&2
  exit 2
fi
if ! "$PYTHON_BIN" - "$TARGET_DELTAK" "$TARGET_EXT_UM" "$CYCLES_MAX" <<'PY'
import math
import sys
for value in map(float, sys.argv[1:]):
    if not math.isfinite(value) or value <= 0.0:
        raise SystemExit(1)
PY
then
  echo "ERROR: TARGET_DELTAK, TARGET_EXT_UM, and CYCLES_MAX must be finite and positive" >&2
  exit 2
fi

SAFE_LABEL=$(printf '%s' "$RUN_LABEL" | tr -cs 'A-Za-z0-9._-' '_')
OUTROOT=${OUTROOT:-$ROOT/runs/v10_2_30_${SAFE_LABEL}_event_growth_v5_100um_$(date +%Y%m%d_%H%M%S)}

export DELTA_K_MPA_SQRT_M="$TARGET_DELTAK"
export OUTROOT TARGET_EXT_UM CYCLES_MAX HAZARD_SEED
export V10230_SAVE_ACTIVE_STATE_SNAPSHOT=${V10230_SAVE_ACTIVE_STATE_SNAPSHOT:-1}
export V10230_HIGH_CYCLE_CHECKPOINT_DIR=${V10230_HIGH_CYCLE_CHECKPOINT_DIR:-$OUTROOT}
export V10230_HIGH_CYCLE_CHECKPOINT_MIN_SECONDS=${V10230_HIGH_CYCLE_CHECKPOINT_MIN_SECONDS:-30}
export V10230_DMD_REUSE_MAX_SEGMENTS=${V10230_DMD_REUSE_MAX_SEGMENTS:-16}
export V10230_DMD_REUSE_GROWTH_FACTOR=${V10230_DMD_REUSE_GROWTH_FACTOR:-2}

printf '%s\n' "v10.2.30 generic weak-T event-to-event fatigue-growth launcher"
printf '  run_label=%s\n' "$RUN_LABEL"
printf '  target_fraction=%s\n' "$TARGET_FRACTION"
printf '  target_DeltaK_MPa_sqrt_m=%s\n' "$TARGET_DELTAK"
printf '  crack_extension_target_um=%s\n' "$TARGET_EXT_UM"
printf '  cycle_censor=%s\n' "$CYCLES_MAX"
printf '  stochastic_threshold=Exp(1) cumulative-hazard draw, seed=%s\n' "$HAZARD_SEED"
printf '  output=%s\n' "$OUTROOT"
printf '  live_checkpoint_dir=%s\n' "$V10230_HIGH_CYCLE_CHECKPOINT_DIR"

# The qualified low-level launcher remains the command source. Generate its
# run-specific metadata copy outside the repository so the low-level clean-tree
# gate does not reject the wrapper's own temporary file. BSD/macOS mktemp
# requires the XXXXXX template at the end of the path.
BASE_LAUNCHER="$ROOT/scripts/run_v10_2_30_weakt_0p55_high_cycle_1e12.sh"
GENERATED_LAUNCHER=$(mktemp "${TMPDIR:-/tmp}/v10_2_30_generic_weakt.XXXXXX")
cleanup_generated_launcher() {
  rm -f -- "$GENERATED_LAUNCHER"
}
trap cleanup_generated_launcher EXIT

"$PYTHON_BIN" - "$BASE_LAUNCHER" "$GENERATED_LAUNCHER" "$RUN_LABEL" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1])
output = Path(sys.argv[2])
run_label = sys.argv[3]
text = source.read_text()
text = text.replace(
    '"schema": "v10.2.30_weakt_0p55_native_high_cycle_1e12_v1",',
    '"schema": "v10.2.30_generic_weakt_event_to_event_growth_driver_v1",',
)
text = text.replace(
    'print("v10.2.30 real weak-T 0.55 high-cycle production run")',
    f'print("v10.2.30 real weak-T {run_label} event-to-event production run")',
)
text = text.replace(
    '"schema": "v10.2.30_weakt_0p55_high_cycle_summary_v1",',
    '"schema": "v10.2.30_generic_weakt_event_growth_summary_v2",',
)
output.write_text(text)
PY
chmod +x "$GENERATED_LAUNCHER"
bash -n "$GENERATED_LAUNCHER"

set +e
bash "$GENERATED_LAUNCHER"
RC=$?
set -e

if [[ ! -d "$OUTROOT" ]]; then
  echo "ERROR: low-level launcher exited with code $RC before creating $OUTROOT" >&2
  echo "exit_code=$RC"
  echo "output=$OUTROOT"
  exit "$RC"
fi

"$PYTHON_BIN" - \
  "$OUTROOT" "$RUN_LABEL" "$TARGET_FRACTION" "$TARGET_DELTAK" \
  "$TARGET_EXT_UM" "$CYCLES_MAX" "$HAZARD_SEED" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_label = sys.argv[2]
target_fraction = sys.argv[3]
target_delta_k = float(sys.argv[4])
target_extension_um = float(sys.argv[5])
cycle_censor = float(sys.argv[6])
hazard_seed = int(sys.argv[7])

stochastic_metadata = {
    "threshold_is_stochastic": True,
    "threshold_distribution": "unit_exponential_in_cumulative_hazard_action",
    "threshold_sampling": "independent_draw_per_first_passage_interval",
    "hazard_seed": hazard_seed,
}

manifest_path = root / "high_cycle_run_manifest.json"
manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
manifest.update(
    {
        "schema": "v10.2.30_generic_weakt_event_to_event_growth_v2",
        "run_label": run_label,
        "target_fraction": target_fraction,
        "target_deltaK_MPa_sqrt_m": target_delta_k,
        "target_crack_extension_um": target_extension_um,
        "cycles_max_censor": cycle_censor,
        "primary_objective": "measure_event_resolved_and_developed_da_dN",
        "cycle_horizon_role": "maximum_censor_not_required_target",
        "generic_launcher": "scripts/run_v10_2_30_weakt_high_cycle_1e12.sh",
        "active_state_snapshot_requested": True,
        "live_checkpointing_requested": True,
        **stochastic_metadata,
    }
)
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

summary_path = root / "high_cycle_summary.json"
summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
summary.update(
    {
        "schema": "v10.2.30_generic_weakt_event_growth_summary_v2",
        "run_label": run_label,
        "target_fraction": target_fraction,
        "target_deltaK_MPa_sqrt_m": target_delta_k,
        "target_crack_extension_um": target_extension_um,
        "cycles_max_censor": cycle_censor,
        **stochastic_metadata,
    }
)

checkpoint_path = root / "high_cycle_live_checkpoint.json"
if checkpoint_path.is_file():
    checkpoint = json.loads(checkpoint_path.read_text())
    stochastic = checkpoint.get("stochastic", {})
    action = stochastic.get("hazard_action_current")
    threshold = stochastic.get("hazard_threshold_action")
    clock = stochastic.get("B")
    if action is not None:
        action = float(action)
        summary["current_interval_physical_hazard_action"] = action
        summary["current_interval_ensemble_event_probability"] = -math.expm1(-action)
    if threshold is not None:
        summary["current_interval_sampled_threshold"] = float(threshold)
    if clock is not None:
        summary["current_interval_normalized_clock_B"] = float(clock)

summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY

if [[ -s "$OUTROOT/high_cycle_live_checkpoint.json" ]]; then
  set +e
  "$PYTHON_BIN" scripts/analyze_v10_2_30_high_cycle_live_checkpoint.py "$OUTROOT" \
    | tee "$OUTROOT/live_high_cycle_visuals_console.log"
  LIVE_VISUAL_RC=${PIPESTATUS[0]}
  set -e
  if [[ "$LIVE_VISUAL_RC" -ne 0 ]]; then
    echo "WARNING: live-checkpoint visuals failed with exit code $LIVE_VISUAL_RC" >&2
  fi
fi

if [[ -s "$OUTROOT/kinetic_tip_cell_audit_v101.json" ]]; then
  set +e
  "$PYTHON_BIN" scripts/analyze_v10_2_30_high_cycle_visuals.py "$OUTROOT" \
    | tee "$OUTROOT/high_cycle_visuals_console.log"
  VISUAL_RC=${PIPESTATUS[0]}
  set -e
  if [[ "$VISUAL_RC" -ne 0 ]]; then
    echo "WARNING: completed-run visual diagnostics failed with exit code $VISUAL_RC" >&2
  fi
fi

if [[ -s "$OUTROOT/steps_0300K.csv" ]]; then
  set +e
  "$PYTHON_BIN" scripts/analyze_v10_2_30_developed_fatigue_growth.py \
    "$OUTROOT" \
    --temperature-K 300 \
    --target-extension-um "$TARGET_EXT_UM" \
    --development-extension-um 20 \
    --stability-window-um 50 \
    --moving-window-um 25 \
    | tee "$OUTROOT/developed_fatigue_growth_console.log"
  GROWTH_RC=${PIPESTATUS[0]}
  set -e
  if [[ "$GROWTH_RC" -ne 0 ]]; then
    echo "WARNING: developed-growth analysis failed with exit code $GROWTH_RC" >&2
  fi
fi

echo "exit_code=$RC"
echo "output=$OUTROOT"
exit "$RC"
