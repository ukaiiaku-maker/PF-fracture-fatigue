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

OUTROOT=${OUTROOT:-runs/v10_2_29_300K_validation_v1}
KERNEL_CACHE_ROOT=${KERNEL_CACHE_ROOT:-runs/v10_2_28_kernel_cache}
FAMILY_JSON=${FAMILY_JSON:-}
MONO_STEPS=${MONO_STEPS:-20000}
CYCLIC_STEPS=${CYCLIC_STEPS:-20000}
TARGET_EXT_UM=${TARGET_EXT_UM:-5}
CYCLIC_CYCLES_MAX=${CYCLIC_CYCLES_MAX:-1e8}
FATIGUE_AMPLITUDE_FRACTION=${FATIGUE_AMPLITUDE_FRACTION:-0.85}
R_RATIO=${R_RATIO:-0.1}
FREQUENCY_HZ=${FREQUENCY_HZ:-1000}
HAZARD_SEED_DBT=${HAZARD_SEED_DBT:-3621}
HAZARD_SEED_WEAKT=${HAZARD_SEED_WEAKT:-1003621}

DBTT_OPTION=v913_paper_dbtt01_0202500_persistent_sites
WEAKT_OPTION=v913_paper_weakT01_0129902_persistent_sites

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

[[ -f "$FAMILY_JSON" ]] || { echo "ERROR: missing FAMILY_JSON=$FAMILY_JSON" >&2; exit 2; }

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

"$PYTHON_BIN" -m pytest -q \
  tests/test_v10_2_29_controller_delegate.py \
  tests/test_v10_2_29_driver_cycle_patch.py \
  tests/test_v10_2_29_event_cycle_accounting.py \
  tests/test_v10_2_29_fatigue_entry_contract.py \
  tests/test_v10_2_29_persistent_cycle_audit.py

common_args=(
  --signed-kernel-family "$FAMILY_JSON"
  --mode 2d
  --temperatures 300
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
  --print-every 200
  --save-snapshots 0
  --no-plots
)

run_monotonic() {
  local entry=$1
  local option=$2
  local seed=$3
  local out=$4
  rm -rf "$out"
  mkdir -p "$out"
  echo "MONOTONIC: entry=$entry option=$option seed=$seed"
  env CLEAVAGE_HAZARD_SEED="$seed" \
    "$PYTHON_BIN" -u -m "$entry" \
      "${common_args[@]}" \
      --parameter-option "$option" \
      --steps "$MONO_STEPS" \
      --dU 2e-7 \
      --adaptive-events --adaptive-event-target 0.15 \
      --out "$out" \
      2>&1 | tee "$out/run.log"
}

DBTT_28="$OUTROOT/monotonic_v10228_dbtt_300K"
DBTT_29="$OUTROOT/monotonic_v10229_dbtt_300K"
WEAKT_28="$OUTROOT/monotonic_v10228_weakt_300K"
WEAKT_29="$OUTROOT/monotonic_v10229_weakt_300K"
CYCLIC="$OUTROOT/cyclic_v10229_weakt_300K"

run_monotonic arrhenius_fracture.sharp_front_v10_2_28_audited \
  "$DBTT_OPTION" "$HAZARD_SEED_DBT" "$DBTT_28"
run_monotonic arrhenius_fracture.sharp_front_v10_2_29_fatigue_audited \
  "$DBTT_OPTION" "$HAZARD_SEED_DBT" "$DBTT_29"
run_monotonic arrhenius_fracture.sharp_front_v10_2_28_audited \
  "$WEAKT_OPTION" "$HAZARD_SEED_WEAKT" "$WEAKT_28"
run_monotonic arrhenius_fracture.sharp_front_v10_2_29_fatigue_audited \
  "$WEAKT_OPTION" "$HAZARD_SEED_WEAKT" "$WEAKT_29"

FATIGUE_DU=$(WEAKT_ROOT="$WEAKT_28" FRACTION="$FATIGUE_AMPLITUDE_FRACTION" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
import numpy as np
root = Path(os.environ["WEAKT_ROOT"])
path = root / "steps_0300K.csv"
if not path.is_file():
    raise SystemExit(f"ERROR: missing monotonic step history: {path}")
data = np.genfromtxt(path, delimiter=",", names=True)
rows = np.atleast_1d(data)
fired = rows[np.asarray(rows["n_fire"]) > 0]
if fired.size == 0:
    raise SystemExit(
        "ERROR: weak-T monotonic reference did not reach first passage; increase MONO_STEPS"
    )
Ucrit = float(np.atleast_1d(fired)[0]["Uapp_m"])
fraction = float(os.environ["FRACTION"])
if not (0.0 < fraction < 1.0):
    raise SystemExit("ERROR: FATIGUE_AMPLITUDE_FRACTION must lie in (0,1)")
print(f"{fraction * Ucrit:.17g}")
PY
)

echo "CYCLIC: weakT 300 K, dU amplitude=$FATIGUE_DU m, R=$R_RATIO, f=$FREQUENCY_HZ Hz"
rm -rf "$CYCLIC"
mkdir -p "$CYCLIC"
env CLEAVAGE_HAZARD_SEED="$HAZARD_SEED_WEAKT" \
  "$PYTHON_BIN" -u -m arrhenius_fracture.sharp_front_v10_2_29_fatigue_audited \
    "${common_args[@]}" \
    --parameter-option "$WEAKT_OPTION" \
    --steps "$CYCLIC_STEPS" \
    --dU "$FATIGUE_DU" \
    --fatigue-cycles \
    --fatigue-hold-load \
    --R "$R_RATIO" \
    --frequency-Hz "$FREQUENCY_HZ" \
    --cycle-block-mode hazard_limited \
    --block-cycles 1e4 \
    --max-block-cycles 1e6 \
    --min-block-cycles 1e-6 \
    --cycles-max "$CYCLIC_CYCLES_MAX" \
    --target-dB 0.10 \
    --target-dN-store 0.10 \
    --target-dN-emit 0.10 \
    --target-dN-mobile 0.10 \
    --target-dN-escape 0.10 \
    --target-dN-peierls 0.10 \
    --target-dN-taylor 0.10 \
    --n-phase 48 \
    --max-da-per-block-um 5 \
    --out "$CYCLIC" \
    2>&1 | tee "$CYCLIC/run.log"

"$PYTHON_BIN" scripts/validate_v10_2_29_300K_outputs.py \
  --dbtt-v10228 "$DBTT_28" \
  --dbtt-v10229 "$DBTT_29" \
  --weakt-v10228 "$WEAKT_28" \
  --weakt-v10229 "$WEAKT_29" \
  --cyclic "$CYCLIC" \
  --out "$OUTROOT/v10_2_29_300K_validation_summary.json"

echo "VALIDATION_COMPLETE: $OUTROOT/v10_2_29_300K_validation_summary.json"
