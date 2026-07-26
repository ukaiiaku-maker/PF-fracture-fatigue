#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

mktemp() {
  if [[ $# -eq 1 && "$1" == *".v10_2_28_four_class_orientation_scheduler.XXXXXX.sh" ]]; then
    local template=$1
    local directory
    local basename_value
    local prefix
    local suffix
    directory=$(dirname "$template")
    basename_value=$(basename "$template")
    prefix=${basename_value%%XXXXXX*}
    suffix=${basename_value#*XXXXXX}
    python - "$directory" "$prefix" "$suffix" <<'PY'
import os
from pathlib import Path
import sys
import tempfile

root = Path(sys.argv[1]).resolve()
prefix = sys.argv[2]
suffix = sys.argv[3]
root.mkdir(parents=True, exist_ok=True)
fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=root)
os.close(fd)
print(path)
PY
    return
  fi
  /usr/bin/mktemp "$@"
}
export -f mktemp

exec bash "$ROOT/scripts/run_v10_2_28_paper_four_class_theta30_1000um.sh" "$@"
