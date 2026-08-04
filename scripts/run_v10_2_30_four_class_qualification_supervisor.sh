#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
exec conda run -n arrhenius-sharp-front-v10-codex python \
  scripts/v10230_qualification_supervisor.py run \
  "${1:-runs/v10_2_30_four_class_qualification}" --max-jobs "${MAX_JOBS:-2}"
