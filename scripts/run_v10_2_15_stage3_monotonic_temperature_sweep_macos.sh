#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE="$ROOT/scripts/run_v10_2_15_stage3_monotonic_temperature_sweep.sh"
PATCHED="$ROOT/scripts/.run_v10_2_15_stage3_monotonic_temperature_sweep_macos.$$.sh"

cleanup() {
  rm -f "$PATCHED"
}
trap cleanup EXIT INT TERM

python - "$SOURCE" "$PATCHED" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1])
out = Path(sys.argv[2])
text = source.read_text()
old_arrays = '''  PIDS=("${new_pids[@]}")
  LABELS=("${new_labels[@]}")
'''
new_arrays = '''  if [[ ${#new_pids[@]} -gt 0 ]]; then
    PIDS=("${new_pids[@]}")
    LABELS=("${new_labels[@]}")
  else
    PIDS=()
    LABELS=()
  fi
'''
if old_arrays not in text:
    raise SystemExit("cannot find scheduler array assignment to patch")
text = text.replace(old_arrays, new_arrays, 1)
old_start = '''  if [[ "$SKIP_FINISHED" == "1" && -f "$case_root/stage3_case_status.json" ]]; then
    echo "SKIP finished: $option T=${temperature}K"
    return 0
  fi

  local cmd=(
'''
new_start = '''  if [[ "$SKIP_FINISHED" == "1" && -f "$case_root/stage3_case_status.json" ]]; then
    echo "SKIP finished: $option T=${temperature}K"
    return 0
  fi
  rm -f "$case_root/RUN_FAILED" "$case_root/exit_code.txt"

  local cmd=(
'''
if old_start not in text:
    raise SystemExit("cannot find case startup block to patch")
text = text.replace(old_start, new_start, 1)
out.write_text(text)
out.chmod(0o755)
PY

set +e
bash "$PATCHED"
rc=$?
set -e
exit "$rc"
