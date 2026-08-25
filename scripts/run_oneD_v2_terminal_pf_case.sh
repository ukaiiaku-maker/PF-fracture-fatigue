#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python}
TEMPERATURE_K=${TEMPERATURE_K:?TEMPERATURE_K is required}
PARAMETER_OPTION=${PARAMETER_OPTION:?PARAMETER_OPTION is required}
HAZARD_SEED=${HAZARD_SEED:?HAZARD_SEED is required}
CASE_OUT=${CASE_OUT:?CASE_OUT is required}
TRANSFER_ROOT=${TRANSFER_ROOT:-/private/tmp/oneD-v2-terminal-predictive-program/analysis_outputs/oneD_v2_terminal_predictive_program}
TRANSFER_REGISTRY=${TRANSFER_REGISTRY:-$TRANSFER_ROOT/oneD_v2_pf_transfer_registry.csv}
TRANSFER_SELECTION=${TRANSFER_SELECTION:-$TRANSFER_ROOT/oneD_v2_pf_transfer_selection.json}
FAMILY_JSON=${FAMILY_JSON:-/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/v10_2_28_kernel_cache/1447653d199f0b43cb475951092d69444c9b785f6fdf518c723792abb3b1f5e5/family.json}

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/private/tmp/oneD-v2-pf-mpl
export PARAMETER_CAMPAIGN=1
export CLEAVAGE_HAZARD_MODE=exponential
export CLEAVAGE_EVENT_LENGTH_MODE=threshold_scaled
export CLEAVAGE_EVENT_MIN_FACTOR=0.5
export CLEAVAGE_EVENT_MAX_FACTOR=4.0
export CLEAVAGE_EVENT_SUBSEGMENT_FRACTION=0.1
export ANISOTROPIC_TRANSPORT_MODE=validated_scalar
export ANISOTROPIC_USE_AVALANCHE_BACKEND=1
export ANISOTROPIC_EMISSION_ENABLED=1
export KERNEL_STRICT_FAMILY_OVERRIDE=1
export SIGNED_KERNEL_FAMILY_JSON="$FAMILY_JSON"
export PERSISTENT_SOURCE_MIN_WIDTH_UM=0
export CLEAVAGE_HAZARD_SEED="$HAZARD_SEED"
export ONED_V2_TRANSFER_REGISTRY="$TRANSFER_REGISTRY"
export ONED_V2_TRANSFER_SELECTION="$TRANSFER_SELECTION"

"$PYTHON_BIN" -u scripts/run_oneD_v2_terminal_pf_transfer.py \
  --signed-kernel-family "$FAMILY_JSON" \
  --mode 2d --temperatures "$TEMPERATURE_K" \
  --nx 36 --ny 72 --dt 8.4 --n-stagger 2 \
  --tip-h-fine 1e-6 --tip-ratio 1.20 --da-phys 5e-6 \
  --target-crack-extension-um 100 \
  --front-state-model moving_pz --tip-source-model continuum \
  --tip-kinetics-mode moving_velocity --bulk-plasticity-mode tip_only \
  --directional-j-mode root_signed --tip-plasticity \
  --active-shielding --signed-active-shielding --mobile-shield-fraction 0 \
  --no-wake-shielding --crystal-aniso --crystal-compete \
  --crystal-theta-deg 0 --crystal-material w --j-decomposition cluster \
  --max-fronts 1 --crack-backend sharp_wake \
  --print-every 200 --save-snapshots 0 --no-plots \
  --parameter-option "$PARAMETER_OPTION" --steps 20000 --dU 2e-7 \
  --adaptive-events --adaptive-event-target 0.15 --out "$CASE_OUT"
