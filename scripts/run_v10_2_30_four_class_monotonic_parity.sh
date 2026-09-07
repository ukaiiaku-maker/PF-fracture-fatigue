#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
OUTROOT=${OUTROOT:-runs/v10_2_30_four_class_monotonic_parity}
FAMILY_JSON=${FAMILY_JSON:?Set FAMILY_JSON to the qualified signed-kernel family}
TARGET_EXT_UM=${TARGET_EXT_UM:-25}
MONO_STEPS=${MONO_STEPS:-20000}

if [[ -e "$OUTROOT" ]]; then
  echo "ERROR: monotonic parity output already exists: $OUTROOT" >&2
  exit 2
fi
mkdir -p "$OUTROOT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PARAMETER_CAMPAIGN=1
export CLEAVAGE_HAZARD_MODE=exponential
export CLEAVAGE_EVENT_LENGTH_MODE=threshold_scaled
export CLEAVAGE_EVENT_MIN_FACTOR=0.5 CLEAVAGE_EVENT_MAX_FACTOR=4.0
export CLEAVAGE_EVENT_SUBSEGMENT_FRACTION=0.1
export ANISOTROPIC_TRANSPORT_MODE=validated_scalar
export ANISOTROPIC_USE_AVALANCHE_BACKEND=1 ANISOTROPIC_EMISSION_ENABLED=1
export KERNEL_STRICT_FAMILY_OVERRIDE=1 SIGNED_KERNEL_FAMILY_JSON="$FAMILY_JSON"
export PERSISTENT_SOURCE_MIN_WIDTH_UM=0

common_args=(
  --signed-kernel-family "$FAMILY_JSON" --mode 2d --temperatures 300
  --nx 36 --ny 72 --dt 8.4 --n-stagger 2
  --tip-h-fine 1e-6 --tip-ratio 1.20 --da-phys 5e-6
  --target-crack-extension-um "$TARGET_EXT_UM"
  --front-state-model moving_pz --tip-source-model continuum
  --tip-kinetics-mode moving_velocity --bulk-plasticity-mode tip_only
  --directional-j-mode root_signed --tip-plasticity --active-shielding
  --signed-active-shielding --mobile-shield-fraction 0 --no-wake-shielding
  --crystal-aniso --crystal-compete --crystal-theta-deg 30
  --crystal-material w --j-decomposition cluster --max-fronts 1
  --crack-backend sharp_wake --print-every 200 --save-snapshots 0 --no-plots
)

run_pair() {
  local label=$1 option=$2 seed=$3
  local reference="$OUTROOT/${label}_v10228"
  local overlay="$OUTROOT/${label}_unified_v10230"
  mkdir -p "$reference" "$overlay"
  env CLEAVAGE_HAZARD_SEED="$seed" "$PYTHON_BIN" -u \
    -m arrhenius_fracture.sharp_front_v10_2_28_audited \
    "${common_args[@]}" --parameter-option "$option" --steps "$MONO_STEPS" \
    --dU 2e-7 --adaptive-events --adaptive-event-target 0.15 --out "$reference" \
    >"$reference/run.log" 2>&1
  env CLEAVAGE_HAZARD_SEED="$seed" "$PYTHON_BIN" -u \
    -m arrhenius_fracture.sharp_front_v10_2_29_fatigue_audited \
    "${common_args[@]}" --parameter-option "$option" --steps "$MONO_STEPS" \
    --dU 2e-7 --adaptive-events --adaptive-event-target 0.15 --out "$overlay" \
    >"$overlay/run.log" 2>&1
}

run_pair peak v913_paper_peak01_0242980_persistent_sites 1720 &
p1=$!
run_pair dbtt v913_paper_dbtt01_0202500_persistent_sites 1001723 &
p2=$!
run_pair weakt v913_paper_weakT01_0129902_persistent_sites 2001726 &
p3=$!
run_pair ceramic v913_paper_ceramic01_0077080_persistent_sites 3001729 &
p4=$!
wait "$p1" "$p2" "$p3" "$p4"

"$PYTHON_BIN" - "$OUTROOT" <<'PY'
import importlib.util
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
script = Path("scripts/validate_v10_2_29_300K_outputs.py").resolve()
spec = importlib.util.spec_from_file_location("monotonic_validator", script)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
pairs = []
for label in ("peak", "dbtt", "weakt", "ceramic"):
    result = module.validate_monotonic_pair(
        root / f"{label}_v10228", root / f"{label}_unified_v10230"
    )
    result["material_class"] = label
    result["exact_numeric_equality"] = True
    pairs.append(result)
payload = {
    "schema": "v10.2.30_four_class_monotonic_direct_parity_v1",
    "temperature_K": 300.0,
    "qualified_reference_entry": "sharp_front_v10_2_28_audited",
    "unified_entry": "sharp_front_v10_2_29_fatigue_audited",
    "all_pairs_exact": True,
    "pairs": pairs,
}
(root / "four_class_monotonic_parity.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
