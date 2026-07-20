#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

EXPECTED_ENV=${EXPECTED_ENV:-arrhenius-sharp-front-v10}
if [[ "${CONDA_DEFAULT_ENV:-}" != "$EXPECTED_ENV" ]]; then
  echo "ERROR: activate conda environment $EXPECTED_ENV" >&2
  exit 2
fi

OUTROOT=${OUTROOT:-runs/v10_2_17_stage3_final_signed_stochastic_500um_theta45_1x_v1}
OUTPUT_DIR=${OUTPUT_DIR:-$OUTROOT/analysis_v10_2_17_temperature_metrics}
FORMATS=${FORMATS:-"png pdf"}
DPI=${DPI:-180}

if [[ ! -d "$OUTROOT" ]]; then
  echo "ERROR: campaign output not found: $OUTROOT" >&2
  exit 2
fi

read -r -a format_words <<< "$FORMATS"
python scripts/plot_v10_2_17_stage3_temperature_metrics.py \
  --outroot "$OUTROOT" \
  --output-dir "$OUTPUT_DIR" \
  --formats "${format_words[@]}" \
  --dpi "$DPI"

echo "Stage 3 plots written to: $OUTPUT_DIR"
