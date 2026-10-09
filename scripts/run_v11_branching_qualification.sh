#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PYTHON_BIN=${PYTHON_BIN:-python}
QUAL_ROOT=${QUAL_ROOT:-$ROOT/runs/v11_branching_qualification_v1}
KERNEL_CACHE_ROOT=${KERNEL_CACHE_ROOT:-$ROOT/runs/v11_branching_direct_kernel_cache_v1}
MAX_JOBS=${MAX_JOBS:-1}
TARGET_EXT_UM=${TARGET_EXT_UM:-25}
STEPS=${STEPS:-2000000}

[[ "$MAX_JOBS" == 1 ]] || { echo "ERROR: qualification is serial (MAX_JOBS=1)" >&2; exit 2; }
[[ ! -e "$QUAL_ROOT" ]] || { echo "ERROR: qualification root exists: $QUAL_ROOT" >&2; exit 2; }
mkdir -p "$QUAL_ROOT" "$KERNEL_CACHE_ROOT"
export KERNEL_CACHE_ROOT

QUAL_ROOT="$QUAL_ROOT" "$PYTHON_BIN" - <<'PY'
from datetime import datetime, timezone
import json, os, pathlib, subprocess
root = pathlib.Path(os.environ["QUAL_ROOT"])
cases = [{"case_id": f"{mode}_theta{theta}_seed3621", "mode": mode, "orientation_deg": theta, "seed": 3621} for theta in (30, 45) for mode in ("control", "mechanistic")]
payload = {"schema": "v11.branching-launcher/1", "campaign_kind": "qualification", "git_head": subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip(), "launcher_pid": os.getppid(), "start_time": datetime.now(timezone.utc).isoformat(), "environment": {"MAX_JOBS": 1, "KERNEL_CACHE_ROOT": os.environ["KERNEL_CACHE_ROOT"]}, "planned_cases": cases}
(root / "launcher.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

for theta in 30 45; do
  for mode in control mechanistic; do
    "$PYTHON_BIN" scripts/run_v11_branching_case.py --mode "$mode" \
      --out "$QUAL_ROOT/${mode}_theta${theta}_seed3621" --orientation "$theta" \
      --seed 3621 --target-um "$TARGET_EXT_UM" --steps "$STEPS"
  done
done

"$PYTHON_BIN" scripts/status_v11_branching_campaign.py "$QUAL_ROOT" > "$QUAL_ROOT/final_status.json"
