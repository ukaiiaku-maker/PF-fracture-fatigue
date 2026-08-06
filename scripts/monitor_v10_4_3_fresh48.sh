#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

CONDA_ENV=${CONDA_ENV:-arrhenius-sharp-front-v10}
OUTROOT=${OUTROOT:-/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/v10_4_3_theta0_rate1x_bulk_PT_positiveJ_plastic_dominance_fresh48_base3621_v1}

if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
  exec python scripts/monitor_v10_4_3_campaign.py \
    --outroot "$OUTROOT" "$@"
fi

command -v conda >/dev/null 2>&1 || {
  echo "ERROR: conda is unavailable and '$CONDA_ENV' is not active" >&2
  exit 2
}

exec conda run --no-capture-output -n "$CONDA_ENV" \
  python scripts/monitor_v10_4_3_campaign.py \
  --outroot "$OUTROOT" "$@"
