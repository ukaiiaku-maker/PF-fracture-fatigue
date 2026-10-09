#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PYTHON_BIN=${PYTHON_BIN:-python}
PROD_ROOT=${PROD_ROOT:-$ROOT/runs/v11_branching_weakt700K_production_tranche_v1}
KERNEL_CACHE_ROOT=${KERNEL_CACHE_ROOT:-$ROOT/runs/v11_branching_direct_kernel_cache_v1}
SELECTED_THETA=${SELECTED_THETA:?set SELECTED_THETA from the qualification/screen result}
MAX_JOBS=${MAX_JOBS:-2}
TARGET_EXT_UM=${TARGET_EXT_UM:-25}
STEPS=${STEPS:-2000000}

[[ ! -e "$PROD_ROOT" ]] || { echo "ERROR: production root exists: $PROD_ROOT" >&2; exit 2; }
mkdir -p "$PROD_ROOT" "$KERNEL_CACHE_ROOT"
export KERNEL_CACHE_ROOT

PROD_ROOT="$PROD_ROOT" SELECTED_THETA="$SELECTED_THETA" MAX_JOBS="$MAX_JOBS" "$PYTHON_BIN" - <<'PY'
from datetime import datetime, timezone
import json, os, pathlib, subprocess
root = pathlib.Path(os.environ["PROD_ROOT"]); selected = float(os.environ["SELECTED_THETA"])
orientations = []
for value in (30.0, selected, selected - 5.0, selected + 5.0):
    if 0.0 < value < 90.0 and value not in orientations: orientations.append(value)
cases = [{"case_id": f"mechanistic_theta{theta:g}_seed{seed}", "mode": "mechanistic", "orientation_deg": theta, "seed": seed} for theta in orientations for seed in (3621, 1003621, 2003621, 3003621, 4003621)]
cases.insert(0, {"case_id": "control_theta30_seed3621", "mode": "control", "orientation_deg": 30.0, "seed": 3621})
payload = {"schema": "v11.branching-launcher/1", "campaign_kind": "production_tranche", "git_head": subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip(), "launcher_pid": os.getppid(), "start_time": datetime.now(timezone.utc).isoformat(), "environment": {"MAX_JOBS": int(os.environ["MAX_JOBS"]), "KERNEL_CACHE_ROOT": os.environ["KERNEL_CACHE_ROOT"]}, "planned_cases": cases}
(root / "launcher.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

export PYTHON_BIN PROD_ROOT TARGET_EXT_UM STEPS
"$PYTHON_BIN" - <<'PY'
import json, os, pathlib, subprocess
root = pathlib.Path(os.environ["PROD_ROOT"]); launcher = json.loads((root / "launcher.json").read_text())
workers = []
for case in launcher["planned_cases"]:
    command = [os.environ["PYTHON_BIN"], "scripts/run_v11_branching_case.py", "--mode", case["mode"], "--out", str(root / case["case_id"]), "--orientation", str(case["orientation_deg"]), "--seed", str(case["seed"]), "--target-um", os.environ["TARGET_EXT_UM"], "--steps", os.environ["STEPS"]]
    workers.append(subprocess.Popen(command))
    if len(workers) >= int(launcher["environment"]["MAX_JOBS"]):
        if workers.pop(0).wait() != 0: raise SystemExit("production case failed")
for worker in workers:
    if worker.wait() != 0: raise SystemExit("production case failed")
PY
