#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
CONDA_ENV=${CONDA_ENV:-arrhenius-sharp-front-v10}
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
  echo "ERROR: activate conda environment '$CONDA_ENV'" >&2
  exit 2
fi

MODE=${MODE:-horizon}
OUTROOT=${OUTROOT:-runs/v10_2_29_high_cycle_${MODE}_300K_v1}
REFERENCE_ROOT=${REFERENCE_ROOT:-runs/v10_2_29_300K_validation_v1/monotonic_v10228_weakt_300K}
KERNEL_CACHE_ROOT=${KERNEL_CACHE_ROOT:-runs/v10_2_28_kernel_cache}
FAMILY_JSON=${FAMILY_JSON:-}
PARAMETER_OPTION=${PARAMETER_OPTION:-v913_paper_weakT01_0129902_persistent_sites}
TEMPERATURE_K=${TEMPERATURE_K:-300}
R_RATIO=${R_RATIO:-0.1}
FREQUENCY_HZ=${FREQUENCY_HZ:-1000}
HAZARD_SEEDS=${HAZARD_SEEDS:-1720}
TARGET_INCREMENT=${TARGET_INCREMENT:-10}
TARGET_DB=${TARGET_DB:-0.1}
MAX_RECORDS_PER_HORIZON=${MAX_RECORDS_PER_HORIZON:-1000}
DEVELOPED_START_UM=${DEVELOPED_START_UM:-25}
DEVELOPED_END_UM=${DEVELOPED_END_UM:-75}
FORCE=${FORCE:-0}

case "$MODE" in
  horizon)
    DELTA_K_FRACTIONS=${DELTA_K_FRACTIONS:-0.25}
    HORIZONS=${HORIZONS:-"1e6 1e9 1e12"}
    TARGET_EXT_UM=${TARGET_EXT_UM:-5}
    STEPS=${STEPS:-2000}
    ;;
  growth)
    DELTA_K_FRACTIONS=${DELTA_K_FRACTIONS:-"0.70 0.85 0.95"}
    HORIZONS=${HORIZONS:-1e12}
    TARGET_EXT_UM=${TARGET_EXT_UM:-100}
    STEPS=${STEPS:-50000}
    ;;
  *)
    echo "ERROR: MODE must be horizon or growth" >&2
    exit 2
    ;;
esac

mkdir -p "$OUTROOT"

"$PYTHON_BIN" scripts/install_v10_2_27_four_class_registry.py
"$PYTHON_BIN" scripts/install_v10_2_27_four_class_registry.py --check-only

if [[ -z "$FAMILY_JSON" ]]; then
  resolution="$OUTROOT/v10_2_29_kernel_resolution.json"
  "$PYTHON_BIN" scripts/ensure_v10_2_28_signed_kernel.py \
    --theta-deg 30 \
    --target-extension-um 20 \
    --branching-mode single_front \
    --maximum-fronts 1 \
    --process-zone-length-um 50 \
    --process-zone-bins 80 \
    --mesh-nx 36 \
    --mesh-ny 72 \
    --tip-h-fine-um 1 \
    --tip-ratio 1.20 \
    --da-phys-um 5 \
    --mode auto \
    --cache-root "$KERNEL_CACHE_ROOT" \
    --json > "$resolution.tmp"
  mv "$resolution.tmp" "$resolution"
  FAMILY_JSON=$(RESOLUTION="$resolution" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
resolution = json.loads(Path(os.environ["RESOLUTION"]).read_text())
family = Path(resolution["family"]).resolve()
if not family.is_file():
    raise SystemExit(f"ERROR: resolved family is missing: {family}")
print(family)
PY
  )
fi

[[ -f "$FAMILY_JSON" ]] || {
  echo "ERROR: missing FAMILY_JSON=$FAMILY_JSON" >&2
  exit 2
}

REFERENCE_STEPS="$REFERENCE_ROOT/steps_$(printf '%04d' "$TEMPERATURE_K")K.csv"
[[ -f "$REFERENCE_STEPS" ]] || {
  echo "ERROR: missing monotonic reference $REFERENCE_STEPS" >&2
  echo "Set REFERENCE_ROOT to the matching 300 K monotonic parameter-option case." >&2
  exit 2
}

DELTA_K_CRIT_MPA=$(REFERENCE_STEPS="$REFERENCE_STEPS" R_RATIO="$R_RATIO" "$PYTHON_BIN" - <<'PY'
import os
import numpy as np
path = os.environ["REFERENCE_STEPS"]
R = float(os.environ["R_RATIO"])
rows = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
fired = rows[np.asarray(rows["n_fire"], float) > 0.0]
if fired.size == 0:
    raise SystemExit("ERROR: monotonic reference has no first-passage event")
Kcrit = float(np.atleast_1d(fired)[0]["KJ_Pa_sqrtm"]) / 1.0e6
print(f"{(1.0 - R) * Kcrit:.17g}")
PY
)

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PARAMETER_CAMPAIGN=1
export CLEAVAGE_HAZARD_MODE=exponential
export CLEAVAGE_EVENT_LENGTH_MODE=threshold_scaled
export CLEAVAGE_EVENT_MIN_FACTOR=${CLEAVAGE_EVENT_MIN_FACTOR:-0.5}
export CLEAVAGE_EVENT_MAX_FACTOR=${CLEAVAGE_EVENT_MAX_FACTOR:-4.0}
export CLEAVAGE_EVENT_SUBSEGMENT_FRACTION=${CLEAVAGE_EVENT_SUBSEGMENT_FRACTION:-0.1}
export ANISOTROPIC_TRANSPORT_MODE=validated_scalar
export ANISOTROPIC_USE_AVALANCHE_BACKEND=1
export ANISOTROPIC_EMISSION_ENABLED=1
export KERNEL_STRICT_FAMILY_OVERRIDE=1
export SIGNED_KERNEL_FAMILY_JSON="$FAMILY_JSON"
export PERSISTENT_SOURCE_MIN_WIDTH_UM=${PERSISTENT_SOURCE_MIN_WIDTH_UM:-0}

common_args=(
  --signed-kernel-family "$FAMILY_JSON"
  --mode 2d
  --temperatures "$TEMPERATURE_K"
  --nx 36 --ny 72
  --dt 8.4 --n-stagger 2
  --tip-h-fine 1e-6 --tip-ratio 1.20
  --da-phys 5e-6
  --target-crack-extension-um "$TARGET_EXT_UM"
  --front-state-model moving_pz
  --tip-source-model continuum
  --tip-kinetics-mode moving_velocity
  --bulk-plasticity-mode tip_only
  --directional-j-mode root_signed
  --tip-plasticity
  --active-shielding
  --signed-active-shielding
  --mobile-shield-fraction 0
  --no-wake-shielding
  --crystal-aniso --crystal-compete
  --crystal-theta-deg 30
  --crystal-material w
  --j-decomposition cluster
  --max-fronts 1
  --crack-backend sharp_wake
  --dU 2e-7
  --fatigue-cycles
  --fatigue-hold-load
  --R "$R_RATIO"
  --frequency-Hz "$FREQUENCY_HZ"
  --cycle-block-mode hazard_limited
  --min-block-cycles 1e-6
  --target-dB "$TARGET_DB"
  --target-dN-store "$TARGET_INCREMENT"
  --target-dN-emit "$TARGET_INCREMENT"
  --target-dN-mobile "$TARGET_INCREMENT"
  --target-dN-escape "$TARGET_INCREMENT"
  --target-dN-peierls "$TARGET_INCREMENT"
  --target-dN-taylor "$TARGET_INCREMENT"
  --n-phase 48
  --max-da-per-block-um 5
  --adaptive-events --adaptive-event-target 0.2
  --print-every 100
  --save-snapshots 0
  --no-plots
  --parameter-option "$PARAMETER_OPTION"
)

printf 'v10.2.29 high-cycle fatigue\n'
printf '  mode=%s option=%s T=%s K R=%s f=%s Hz\n' \
  "$MODE" "$PARAMETER_OPTION" "$TEMPERATURE_K" "$R_RATIO" "$FREQUENCY_HZ"
printf '  monotonic DeltaKcrit=%s MPa*sqrt(m)\n' "$DELTA_K_CRIT_MPA"
printf '  fractions=%s horizons=%s seeds=%s\n' \
  "$DELTA_K_FRACTIONS" "$HORIZONS" "$HAZARD_SEEDS"

for FRACTION in $DELTA_K_FRACTIONS; do
  DELTA_K=$(
    FRACTION="$FRACTION" DELTA_K_CRIT_MPA="$DELTA_K_CRIT_MPA" \
    "$PYTHON_BIN" - <<'PY'
import os
fraction = float(os.environ["FRACTION"])
critical = float(os.environ["DELTA_K_CRIT_MPA"])
if not (0.0 < fraction <= 1.0):
    raise SystemExit("ERROR: every DeltaK fraction must lie in (0, 1]")
print(f"{fraction * critical:.17g}")
PY
  )
  FTAG=$(printf '%g' "$FRACTION" | tr '.' 'p')
  for HORIZON in $HORIZONS; do
    HTAG=$(printf '%g' "$HORIZON" | tr '+.' 'pp')
    for SEED in $HAZARD_SEEDS; do
      CASE_OUT="$OUTROOT/fraction_${FTAG}/horizon_${HTAG}/seed_${SEED}"
      if [[ "$FORCE" == 1 ]]; then
        rm -rf "$CASE_OUT"
      fi
      if [[ -f "$CASE_OUT/v10_2_29_fixed_deltaK_control.json" ]]; then
        echo "SKIP existing case: $CASE_OUT"
        continue
      fi
      mkdir -p "$CASE_OUT"
      echo
      echo "CASE: fraction=$FRACTION DeltaK=$DELTA_K horizon=$HORIZON seed=$SEED"
      env CLEAVAGE_HAZARD_SEED="$SEED" \
        "$PYTHON_BIN" -u -m \
          arrhenius_fracture.sharp_front_v10_2_29_fixed_deltaK \
          "${common_args[@]}" \
          --target-deltaK-MPa-sqrt-m "$DELTA_K" \
          --steps "$STEPS" \
          --cycles-max "$HORIZON" \
          --block-cycles "$HORIZON" \
          --max-block-cycles "$HORIZON" \
          --out "$CASE_OUT" \
          2>&1 | tee "$CASE_OUT/run.log"

      if [[ "$MODE" == growth ]]; then
        "$PYTHON_BIN" scripts/extract_v10_2_29_developed_fatigue_growth.py \
          "$CASE_OUT" \
          --temperature-K "$TEMPERATURE_K" \
          --developed-start-um "$DEVELOPED_START_UM" \
          --developed-end-um "$DEVELOPED_END_UM" \
          > "$CASE_OUT/developed_growth.log"
      fi
    done
  done
done

if [[ "$MODE" == horizon ]]; then
  "$PYTHON_BIN" scripts/analyze_v10_2_29_horizon_scaling.py \
    "$OUTROOT" \
    --temperature-K "$TEMPERATURE_K" \
    --max-records-per-case "$MAX_RECORDS_PER_HORIZON" \
    --require-censored-horizon
else
  "$PYTHON_BIN" scripts/analyze_v10_2_29_developed_fatigue_campaign.py \
    "$OUTROOT"
fi

echo "HIGH_CYCLE_COMPLETE: $OUTROOT"
