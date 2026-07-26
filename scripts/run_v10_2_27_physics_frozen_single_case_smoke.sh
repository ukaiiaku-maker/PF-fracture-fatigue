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

FAMILY_JSON=${FAMILY_JSON:-$ROOT/runtime_inputs/v10_2_17/v10_2_14_active_only_campaign_family.json}
REGISTRY=${REGISTRY:-$ROOT/arrhenius_fracture/data/materials/v10_2_27_paper_four_class_registry.csv}
OPTION=${OPTION:-v913_paper_peak01_0242980_persistent_sites}
TEMPERATURE_K=${TEMPERATURE_K:-700}
HAZARD_SEED=${HAZARD_SEED:-3621}
TARGET_EXT_UM=${TARGET_EXT_UM:-25}
STEPS=${STEPS:-200000}
THETA=${THETA:-30}
OUT=${OUT:-$ROOT/runs/v10_2_27_physics_frozen_peak700_theta30_seed3621_25um_v1}
EXPECTED_FAMILY_SHA256=${EXPECTED_FAMILY_SHA256:-}

for required in "$FAMILY_JSON" "$REGISTRY"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: missing required input: $required" >&2
    exit 2
  fi
done

FAMILY_SHA256=$(
  "$PYTHON_BIN" - "$FAMILY_JSON" <<'PY'
import hashlib
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
print(hashlib.sha256(path.read_bytes()).hexdigest())
PY
)

if [[ -n "$EXPECTED_FAMILY_SHA256" && "$FAMILY_SHA256" != "$EXPECTED_FAMILY_SHA256" ]]; then
  echo "ERROR: signed-kernel family SHA mismatch" >&2
  echo "Expected: $EXPECTED_FAMILY_SHA256" >&2
  echo "Actual:   $FAMILY_SHA256" >&2
  exit 2
fi

rm -rf "$OUT"
mkdir -p "$OUT"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PARAMETER_CAMPAIGN=1
export CLEAVAGE_HAZARD_MODE=exponential
export CLEAVAGE_HAZARD_SEED="$HAZARD_SEED"
export CLEAVAGE_EVENT_LENGTH_MODE=threshold_scaled
export CLEAVAGE_EVENT_MIN_FACTOR=0.5
export CLEAVAGE_EVENT_MAX_FACTOR=4.0
export CLEAVAGE_EVENT_SUBSEGMENT_FRACTION=0.1
export ANISOTROPIC_TRANSPORT_MODE=validated_scalar
export ANISOTROPIC_USE_AVALANCHE_BACKEND=1
export ANISOTROPIC_EMISSION_ENABLED=1
export PERSISTENT_SOURCE_MIN_WIDTH_UM=0

"$PYTHON_BIN" - "$OUT" "$FAMILY_JSON" "$FAMILY_SHA256" "$REGISTRY" "$OPTION" "$TEMPERATURE_K" "$HAZARD_SEED" "$TARGET_EXT_UM" "$THETA" <<'PY'
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
payload = {
    "schema": "v10.2.27_physics_frozen_single_case_smoke_v1",
    "signed_kernel_family": str(pathlib.Path(sys.argv[2]).resolve()),
    "signed_kernel_family_sha256": sys.argv[3],
    "parameter_registry": str(pathlib.Path(sys.argv[4]).resolve()),
    "parameter_option": sys.argv[5],
    "temperature_K": float(sys.argv[6]),
    "hazard_seed": int(sys.argv[7]),
    "target_extension_um": float(sys.argv[8]),
    "theta_deg": float(sys.argv[9]),
    "stochastic_cleavage_first_passage": True,
    "cleavage_hazard_mode": "exponential",
    "cleavage_event_length_mode": "threshold_scaled",
    "cleavage_event_min_factor": 0.5,
    "cleavage_event_max_factor": 4.0,
    "cleavage_event_subsegment_fraction": 0.1,
    "front_state_model": "moving_pz",
    "tip_kinetics_mode": "moving_velocity",
    "process_zone_microstructure_advected": True,
    "active_shielding": True,
    "signed_active_shielding": True,
    "wake_shielding": False,
    "mechanics_or_kinetic_parameters_modified": False,
}
(out / "physics_frozen_smoke_contract.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY

cmd=(
  "$PYTHON_BIN" -u -m arrhenius_fracture.sharp_front_v10_2_27_audited
  --signed-kernel-family "$FAMILY_JSON"
  --mode 2d
  --parameter-registry "$REGISTRY"
  --parameter-option "$OPTION"
  --temperatures "$TEMPERATURE_K"
  --steps "$STEPS"
  --nx 36 --ny 72
  --dU 2e-7 --dt 8.4 --n-stagger 2
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
  --crystal-theta-deg "$THETA"
  --crystal-material w
  --j-decomposition cluster
  --max-fronts 1
  --crack-backend sharp_wake
  --adaptive-events --adaptive-event-target 0.15
  --print-every 100
  --save-snapshots 4
  --snapshot-cols 2
  --out "$OUT"
)

{
  echo '#!/usr/bin/env bash'
  printf 'CLEAVAGE_HAZARD_MODE=%q ' "$CLEAVAGE_HAZARD_MODE"
  printf 'CLEAVAGE_HAZARD_SEED=%q ' "$CLEAVAGE_HAZARD_SEED"
  printf 'CLEAVAGE_EVENT_LENGTH_MODE=%q ' "$CLEAVAGE_EVENT_LENGTH_MODE"
  printf 'CLEAVAGE_EVENT_MIN_FACTOR=%q ' "$CLEAVAGE_EVENT_MIN_FACTOR"
  printf 'CLEAVAGE_EVENT_MAX_FACTOR=%q ' "$CLEAVAGE_EVENT_MAX_FACTOR"
  printf 'CLEAVAGE_EVENT_SUBSEGMENT_FRACTION=%q ' "$CLEAVAGE_EVENT_SUBSEGMENT_FRACTION"
  printf '%q ' "${cmd[@]}"
  printf '\n'
} > "$OUT/command.sh"
chmod +x "$OUT/command.sh"

"${cmd[@]}" 2>&1 | tee "$OUT/run.log"

echo "Physics-frozen smoke completed"
echo "Output: $OUT"
echo "Family SHA256: $FAMILY_SHA256"
