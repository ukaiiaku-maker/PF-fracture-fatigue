#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
TARGET_DELTAK=${TARGET_DELTAK:-${DELTA_K_MPA_SQRT_M:-}}
TARGET_FRACTION=${TARGET_FRACTION:-custom}
RUN_LABEL=${RUN_LABEL:-weakt_${TARGET_FRACTION}}

if [[ -z "$TARGET_DELTAK" ]]; then
  echo "ERROR: set TARGET_DELTAK in MPa*sqrt(m)" >&2
  exit 2
fi
if ! "$PYTHON_BIN" - "$TARGET_DELTAK" <<'PY'
import math
import sys
value = float(sys.argv[1])
if not math.isfinite(value) or value <= 0.0:
    raise SystemExit(1)
PY
then
  echo "ERROR: TARGET_DELTAK must be a finite positive number" >&2
  exit 2
fi

SAFE_LABEL=$(printf '%s' "$RUN_LABEL" | tr -cs 'A-Za-z0-9._-' '_')
OUTROOT=${OUTROOT:-$ROOT/runs/v10_2_30_${SAFE_LABEL}_high_cycle_v4_1e12_$(date +%Y%m%d_%H%M%S)}

export DELTA_K_MPA_SQRT_M="$TARGET_DELTAK"
export OUTROOT
export V10230_SAVE_ACTIVE_STATE_SNAPSHOT=${V10230_SAVE_ACTIVE_STATE_SNAPSHOT:-1}

printf '%s\n' "v10.2.30 generic weak-T high-cycle launcher"
printf '  run_label=%s\n' "$RUN_LABEL"
printf '  target_fraction=%s\n' "$TARGET_FRACTION"
printf '  target_DeltaK_MPa_sqrt_m=%s\n' "$TARGET_DELTAK"
printf '  output=%s\n' "$OUTROOT"
printf '  signed_MPZ_snapshot=%s\n' "$V10230_SAVE_ACTIVE_STATE_SNAPSHOT"

set +e
bash scripts/run_v10_2_30_weakt_0p55_high_cycle_1e12.sh
RC=$?
set -e

"$PYTHON_BIN" - "$OUTROOT" "$RUN_LABEL" "$TARGET_FRACTION" "$TARGET_DELTAK" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_label = sys.argv[2]
target_fraction = sys.argv[3]
target_delta_k = float(sys.argv[4])

manifest_path = root / "high_cycle_run_manifest.json"
manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
manifest.update(
    {
        "schema": "v10.2.30_generic_weakt_native_high_cycle_v1",
        "run_label": run_label,
        "target_fraction": target_fraction,
        "target_deltaK_MPa_sqrt_m": target_delta_k,
        "generic_launcher": "scripts/run_v10_2_30_weakt_high_cycle_1e12.sh",
        "active_state_snapshot_requested": True,
    }
)
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

summary_path = root / "high_cycle_summary.json"
summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
summary.update(
    {
        "schema": "v10.2.30_generic_weakt_high_cycle_summary_v1",
        "run_label": run_label,
        "target_fraction": target_fraction,
        "target_deltaK_MPa_sqrt_m": target_delta_k,
    }
)
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY

if [[ -s "$OUTROOT/kinetic_tip_cell_audit_v101.json" ]]; then
  set +e
  "$PYTHON_BIN" scripts/analyze_v10_2_30_high_cycle_visuals.py "$OUTROOT" \
    | tee "$OUTROOT/high_cycle_visuals_console.log"
  VISUAL_RC=${PIPESTATUS[0]}
  set -e
  if [[ "$VISUAL_RC" -ne 0 ]]; then
    echo "WARNING: visual diagnostics failed with exit code $VISUAL_RC" >&2
  fi
fi

echo "exit_code=$RC"
echo "output=$OUTROOT"
exit "$RC"
