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

REFERENCE_ROOT=${REFERENCE_ROOT:-runs/v10_2_29_300K_validation_v1}
OUTROOT=${OUTROOT:-runs/v10_2_29_300K_block_convergence_v1}
TARGET_LEVELS=${TARGET_LEVELS:-"0.1 1 10"}
CYCLES_MAX=${CYCLES_MAX:-5}
STEPS=${STEPS:-40000}
R_RATIO=${R_RATIO:-0.1}
FREQUENCY_HZ=${FREQUENCY_HZ:-1000}
FATIGUE_AMPLITUDE_FRACTION=${FATIGUE_AMPLITUDE_FRACTION:-0.85}
HAZARD_SEED=${HAZARD_SEED:-1003621}
FAMILY_JSON=${FAMILY_JSON:-}
WEAKT_OPTION=v913_paper_weakT01_0129902_persistent_sites

mkdir -p "$OUTROOT"

"$PYTHON_BIN" scripts/install_v10_2_27_four_class_registry.py
"$PYTHON_BIN" scripts/install_v10_2_27_four_class_registry.py --check-only

if [[ -z "$FAMILY_JSON" ]]; then
  resolution="$REFERENCE_ROOT/v10_2_29_kernel_resolution.json"
  [[ -f "$resolution" ]] || {
    echo "ERROR: missing kernel resolution from validated run: $resolution" >&2
    exit 2
  }
  FAMILY_JSON=$(RESOLUTION="$resolution" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
payload = json.loads(Path(os.environ["RESOLUTION"]).read_text())
family = Path(payload["family"]).resolve()
if not family.is_file():
    raise SystemExit(f"ERROR: signed kernel family is missing: {family}")
print(family)
PY
  )
fi
[[ -f "$FAMILY_JSON" ]] || { echo "ERROR: missing FAMILY_JSON=$FAMILY_JSON" >&2; exit 2; }

WEAKT_REFERENCE="$REFERENCE_ROOT/monotonic_v10228_weakt_300K/steps_0300K.csv"
[[ -f "$WEAKT_REFERENCE" ]] || {
  echo "ERROR: missing weak-T monotonic reference: $WEAKT_REFERENCE" >&2
  exit 2
}

FATIGUE_DU=$(REFERENCE="$WEAKT_REFERENCE" FRACTION="$FATIGUE_AMPLITUDE_FRACTION" "$PYTHON_BIN" - <<'PY'
import os
import numpy as np
from pathlib import Path
path = Path(os.environ["REFERENCE"])
rows = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
fired = rows[np.asarray(rows["n_fire"], float) > 0.0]
if fired.size == 0:
    raise SystemExit(f"ERROR: no monotonic first-passage row in {path}")
fraction = float(os.environ["FRACTION"])
if not (0.0 < fraction < 1.0):
    raise SystemExit("ERROR: FATIGUE_AMPLITUDE_FRACTION must lie in (0,1)")
print(f"{fraction * float(np.atleast_1d(fired)[0]['Uapp_m']):.17g}")
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
  --temperatures 300
  --nx 36 --ny 72
  --dt 8.4 --n-stagger 2
  --tip-h-fine 1e-6 --tip-ratio 1.20
  --da-phys 5e-6
  --target-crack-extension-um 5
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
  --print-every 2000
  --save-snapshots 0
  --no-plots
)

read -r -a target_array <<< "$TARGET_LEVELS"
for target in "${target_array[@]}"; do
  tag=${target//-/m}
  tag=${tag//./p}
  tag=${tag//+/}
  run="$OUTROOT/target_$tag"
  rm -rf "$run"
  mkdir -p "$run"
  echo "TARGET=$target cycles, physical exposure=$CYCLES_MAX cycles, dU=$FATIGUE_DU m"
  env CLEAVAGE_HAZARD_SEED="$HAZARD_SEED" \
    "$PYTHON_BIN" -u -m arrhenius_fracture.sharp_front_v10_2_29_fatigue_audited \
      "${common_args[@]}" \
      --parameter-option "$WEAKT_OPTION" \
      --steps "$STEPS" \
      --dU "$FATIGUE_DU" \
      --fatigue-cycles \
      --fatigue-hold-load \
      --R "$R_RATIO" \
      --frequency-Hz "$FREQUENCY_HZ" \
      --cycle-block-mode hazard_limited \
      --block-cycles 1e4 \
      --max-block-cycles 1e6 \
      --min-block-cycles 1e-6 \
      --cycles-max "$CYCLES_MAX" \
      --target-dB "$target" \
      --target-dN-store "$target" \
      --target-dN-emit "$target" \
      --target-dN-mobile "$target" \
      --target-dN-escape "$target" \
      --target-dN-peierls "$target" \
      --target-dN-taylor "$target" \
      --n-phase 48 \
      --max-da-per-block-um 5 \
      --out "$run" \
      2>&1 | tee "$run/run.log"

  "$PYTHON_BIN" scripts/summarize_v10_2_29_cycle_blocks.py "$run" \
    > "$run/cycle_block_summary.log"
done

"$PYTHON_BIN" scripts/compare_v10_2_29_block_convergence.py "$OUTROOT" \
  --targets "${target_array[@]}" \
  --minimum-cycles "$CYCLES_MAX" \
  --out "$OUTROOT/block_target_convergence.json" \
  > "$OUTROOT/block_target_convergence.log"

echo "BLOCK_CONVERGENCE_COMPLETE: $OUTROOT/block_target_convergence.json"
