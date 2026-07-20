#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE="$ROOT_DIR/scripts/run_v10_2_15_stage3_monotonic_temperature_sweep.sh"
GENERATED="$ROOT_DIR/scripts/.run_v10_2_16_generated_$$.sh"

export OUTROOT=${OUTROOT:-runs/v10_2_16_stage3_continuum_four_option_100um_theta45_v1}
export TARGET_EXT_UM=${TARGET_EXT_UM:-100}
export PRINT_EVERY=${PRINT_EVERY:-25}
export PYTHONUNBUFFERED=1

python - "$SOURCE" "$GENERATED" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1])
target = Path(sys.argv[2])
text = source.read_text()
replacements = {
    "arrhenius_fracture.sharp_front_v10_2_15":
        "arrhenius_fracture.sharp_front_v10_2_16",
    "Stage 3 existing-2D parameter-overlay plan":
        "v10.2.16 Stage 3 continuum-source parameter-overlay plan",
    "physics:       unchanged; exact material-manifest overlay only":
        "physics:       accepted 2-D continuum source; exact manifest overlay only",
    "entry=v10.1.7.5": "entry=v10.2.16",
    "Stage 3 runner finished:": "v10.2.16 Stage 3 runner finished:",
}
for old, new in replacements.items():
    if old not in text:
        raise SystemExit(f"upstream launcher changed; missing expected text: {old!r}")
    text = text.replace(old, new)

old_skip = '''  mkdir -p "$case_root"
  if [[ "$SKIP_FINISHED" == "1" && -f "$case_root/stage3_case_status.json" ]]; then
    echo "SKIP finished: $option T=${temperature}K"
    return 0
  fi
  rm -f "$case_root/RUN_FAILED" "$case_root/exit_code.txt"
'''
new_skip = '''  mkdir -p "$case_root"
  if [[ "$SKIP_FINISHED" == "1" && -f "$case_root/stage3_case_status.json" ]]; then
    if "$PYTHON_BIN" - "$case_root/stage3_case_status.json" "$TARGET_EXT_UM" <<'PY_STATUS'
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
target = float(sys.argv[2])
try:
    payload = json.loads(path.read_text())
    complete = payload.get("complete") is True
    status = payload.get("status") == "complete_target_extension"
    recorded_target = float(payload.get("target_extension_um"))
    target_matches = math.isclose(recorded_target, target, rel_tol=0.0, abs_tol=1.0e-9)
except Exception:
    complete = status = target_matches = False
raise SystemExit(0 if complete and status and target_matches else 1)
PY_STATUS
    then
      echo "SKIP complete: $option T=${temperature}K target=${TARGET_EXT_UM}um"
      return 0
    fi
  fi

  # A solver interrupted by a full disk cannot be resumed from its partially
  # written arrays. Preserve a compact diagnostic tail, then restart that case
  # from a clean directory. Verified complete cases returned above are untouched.
  if find "$case_root" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    local interrupted_logs="$OUTROOT/interrupted_case_logs"
    local interrupted_log="$interrupted_logs/${option}_T${temperature}K.log"
    mkdir -p "$interrupted_logs"
    {
      echo "interrupted_case=$option/T${temperature}K"
      echo "restart_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      echo "previous_case_root=$case_root"
      if [[ -f "$log" ]]; then
        echo "--- final 200 lines of previous run.log ---"
        tail -n 200 "$log"
      fi
    } > "$interrupted_log"
    echo "RESTART clean: $option T=${temperature}K; diagnostic=$interrupted_log"
    rm -rf "$case_root"
    mkdir -p "$case_root"
  fi
  rm -f "$case_root/RUN_FAILED" "$case_root/exit_code.txt"
'''
if old_skip not in text:
    raise SystemExit("upstream launcher changed; restart/skip block not found")
text = text.replace(old_skip, new_skip, 1)

old_run = '''  "${cmd[@]}" > "$log" 2>&1
  rc=$?
'''
new_run = '''  "${cmd[@]}" 2>&1 | "$PYTHON_BIN" -u -c '\''
import sys
prefix = sys.argv[1]
for line in sys.stdin:
    print(f"[{prefix}] {line}", end="", flush=True)
'\'' "$option/T${temperature}K" | tee "$log"
  rc=${PIPESTATUS[0]}
'''
if old_run not in text:
    raise SystemExit("upstream launcher changed; solver redirection block not found")
text = text.replace(old_run, new_run, 1)
target.write_text(text)
target.chmod(0o755)
PY

set +e
bash "$GENERATED"
rc=$?
set -e
rm -f "$GENERATED"
exit "$rc"
