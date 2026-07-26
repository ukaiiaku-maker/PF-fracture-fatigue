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

: "${V10227_KERNEL_CONFIGURATION:?resolver must set V10227_KERNEL_CONFIGURATION}"
: "${V10227_KERNEL_CONFIGURATION_FINGERPRINT:?resolver must set V10227_KERNEL_CONFIGURATION_FINGERPRINT}"
: "${V10227_KERNEL_CAPTURE_OUTROOT:?builder must set V10227_KERNEL_CAPTURE_OUTROOT}"
: "${V10227_KERNEL_CAPTURE_REQUIRED_MAX_EXTENSION_UM:?builder must set V10227_KERNEL_CAPTURE_REQUIRED_MAX_EXTENSION_UM}"
: "${V10227_KERNEL_CAPTURE_TEMPERATURE_K:?builder must set V10227_KERNEL_CAPTURE_TEMPERATURE_K}"
: "${KERNEL_CAPTURE_SEED_FAMILY:?set KERNEL_CAPTURE_SEED_FAMILY to an accepted signed-kernel family used only to evolve the production trajectory}"
: "${KERNEL_CAPTURE_PARAMETER_OPTION:?set KERNEL_CAPTURE_PARAMETER_OPTION to the accepted production material option}"
: "${KERNEL_CAPTURE_HAZARD_SEED:?set KERNEL_CAPTURE_HAZARD_SEED explicitly for reproducibility}"

CONFIG=$(cd "$(dirname "$V10227_KERNEL_CONFIGURATION")" && pwd)/$(basename "$V10227_KERNEL_CONFIGURATION")
SNAPSHOT_ROOT=$(cd "$(dirname "$V10227_KERNEL_CAPTURE_OUTROOT")" && pwd)/$(basename "$V10227_KERNEL_CAPTURE_OUTROOT")
SEED_FAMILY=$(cd "$(dirname "$KERNEL_CAPTURE_SEED_FAMILY")" && pwd)/$(basename "$KERNEL_CAPTURE_SEED_FAMILY")
REGISTRY=${KERNEL_CAPTURE_PARAMETER_REGISTRY:-$ROOT/arrhenius_fracture/data/materials/v10_2_27_v913_four_class_paper_registry.csv}
STEPS=${KERNEL_CAPTURE_STEPS:-2000000}
D_U=${KERNEL_CAPTURE_DU_M:-2e-7}
DT=${KERNEL_CAPTURE_DT_S:-8.4}
N_STAGGER=${KERNEL_CAPTURE_N_STAGGER:-2}
EVENT_MINIMUM_FACTOR=${CLEAVAGE_EVENT_MIN_FACTOR:-0.5}
EVENT_MAXIMUM_FACTOR=${CLEAVAGE_EVENT_MAX_FACTOR:-4.0}
EVENT_SUBSEGMENT_FRACTION=${CLEAVAGE_EVENT_SUBSEGMENT_FRACTION:-0.1}
PERSISTENT_SOURCE_MIN_WIDTH_UM=${PERSISTENT_SOURCE_MIN_WIDTH_UM:-0}
EXPECTED_SEED_SHA256=${KERNEL_CAPTURE_SEED_FAMILY_EXPECTED_SHA256:-}

for required in "$CONFIG" "$SEED_FAMILY" "$REGISTRY"; do
  [[ -f "$required" ]] || { echo "ERROR: missing required input: $required" >&2; exit 2; }
done

"$PYTHON_BIN" scripts/install_v10_2_27_four_class_registry.py
"$PYTHON_BIN" scripts/install_v10_2_27_four_class_registry.py --check-only

CACHE_DIR=${V10227_KERNEL_CACHE_DIR:-$(dirname "$SNAPSHOT_ROOT")}
STATE_TABLE="$CACHE_DIR/accepted_production_capture_state_table.csv"
RUN_ROOT="$CACHE_DIR/accepted_production_capture_run"
rm -rf "$RUN_ROOT"
mkdir -p "$RUN_ROOT" "$(dirname "$STATE_TABLE")"

read -r THETA NX NY TIP_H TIP_RATIO DA_M PZ_UM PZ_BINS MIN_PZ_ELEMENTS < <(
  "$PYTHON_BIN" - "$CONFIG" <<'PY'
import json
import sys
p = json.loads(open(sys.argv[1]).read())
print(
    p["theta_deg"],
    p["mesh_nx"],
    p["mesh_ny"],
    p["tip_h_fine_m"],
    p["tip_ratio"],
    p["da_phys_m"],
    1.0e6 * p["process_zone_length_m"],
    p["process_zone_bins"],
    p["minimum_elements_per_process_zone"],
)
PY
)

"$PYTHON_BIN" - "$REGISTRY" "$KERNEL_CAPTURE_PARAMETER_OPTION" "$PZ_UM" "$PZ_BINS" <<'PY'
import csv
import math
import sys
path, option, expected_length_um, expected_bins = sys.argv[1:]
with open(path, newline="") as stream:
    rows = list(csv.DictReader(stream))
matches = [row for row in rows if row.get("option_key") == option]
if len(matches) != 1:
    raise SystemExit(
        f"accepted production option must occur exactly once in registry: {option!r}; "
        f"matches={len(matches)}"
    )
row = matches[0]
length_um = float(row["L_pz_um_recommended"])
bins = int(float(row["n_bins_recommended"]))
if not math.isclose(length_um, float(expected_length_um), rel_tol=0.0, abs_tol=1.0e-12):
    raise SystemExit(
        f"capture option process-zone length mismatch: {length_um} != {expected_length_um} um"
    )
if bins != int(expected_bins):
    raise SystemExit(
        f"capture option process-zone bins mismatch: {bins} != {expected_bins}"
    )
PY

"$PYTHON_BIN" scripts/write_v10_2_27_production_capture_state_table.py \
  --mechanical-config "$CONFIG" \
  --required-max-extension-um "$V10227_KERNEL_CAPTURE_REQUIRED_MAX_EXTENSION_UM" \
  --temperature-K "$V10227_KERNEL_CAPTURE_TEMPERATURE_K" \
  --event-minimum-factor "$EVENT_MINIMUM_FACTOR" \
  --event-maximum-factor "$EVENT_MAXIMUM_FACTOR" \
  --output "$STATE_TABLE"

read -r MAX_ANCHOR_UM PROJECTED_STOP_UM REQUIRED_SEED_PATH_UM < <(
  "$PYTHON_BIN" - "$STATE_TABLE" "$THETA" "$DA_M" <<'PY'
import csv
import math
import sys
path, theta, da_m = sys.argv[1:]
with open(path, newline="") as stream:
    rows = list(csv.DictReader(stream))
maximum_um = max(float(row["cumulative_crack_path_extension_m"]) for row in rows) * 1.0e6
cosine = abs(math.cos(math.radians(float(theta))))
if cosine <= 1.0e-12:
    raise SystemExit("capture theta has zero projected-extension cosine")
projected_stop_um = maximum_um * cosine + 4.0 * float(da_m) * 1.0e6
required_seed_path_um = projected_stop_um / cosine
print(maximum_um, projected_stop_um, required_seed_path_um)
PY
)

SEED_FAMILY_SHA256=$(
  "$PYTHON_BIN" - "$SEED_FAMILY" <<'PY'
import hashlib
import pathlib
import sys
print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
PY
)
if [[ -n "$EXPECTED_SEED_SHA256" && "$SEED_FAMILY_SHA256" != "$EXPECTED_SEED_SHA256" ]]; then
  echo "ERROR: capture bootstrap family SHA-256 mismatch" >&2
  echo "Expected: $EXPECTED_SEED_SHA256" >&2
  echo "Actual:   $SEED_FAMILY_SHA256" >&2
  exit 2
fi

"$PYTHON_BIN" scripts/check_v10_2_27_capture_seed_family.py \
  --family "$SEED_FAMILY" \
  --required-path-extension-um "$REQUIRED_SEED_PATH_UM" \
  --output "$RUN_ROOT/capture_seed_family_audit.json"

export V10227_KERNEL_CAPTURE_SEED_FAMILY="$SEED_FAMILY"
export V10227_KERNEL_CAPTURE_SEED_FAMILY_SHA256="$SEED_FAMILY_SHA256"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PARAMETER_CAMPAIGN=1
export CLEAVAGE_HAZARD_MODE=exponential
export CLEAVAGE_HAZARD_SEED="$KERNEL_CAPTURE_HAZARD_SEED"
export CLEAVAGE_EVENT_LENGTH_MODE=threshold_scaled
export CLEAVAGE_EVENT_MIN_FACTOR="$EVENT_MINIMUM_FACTOR"
export CLEAVAGE_EVENT_MAX_FACTOR="$EVENT_MAXIMUM_FACTOR"
export CLEAVAGE_EVENT_SUBSEGMENT_FRACTION="$EVENT_SUBSEGMENT_FRACTION"
export ANISOTROPIC_TRANSPORT_MODE=validated_scalar
export ANISOTROPIC_USE_AVALANCHE_BACKEND=1
export ANISOTROPIC_EMISSION_ENABLED=1
export PERSISTENT_SOURCE_MIN_WIDTH_UM

cmd=(
  "$PYTHON_BIN" -u -m arrhenius_fracture.sharp_front_v10_2_13_capture
  --atlas-state-table "$STATE_TABLE"
  --atlas-outroot "$SNAPSHOT_ROOT"
  --minimum-elements-per-process-zone "$MIN_PZ_ELEMENTS"
  --signed-kernel-family "$SEED_FAMILY"
  --mode 2d
  --parameter-registry "$REGISTRY"
  --parameter-option "$KERNEL_CAPTURE_PARAMETER_OPTION"
  --temperatures "$V10227_KERNEL_CAPTURE_TEMPERATURE_K"
  --steps "$STEPS"
  --nx "$NX" --ny "$NY"
  --dU "$D_U" --dt "$DT" --n-stagger "$N_STAGGER"
  --tip-h-fine "$TIP_H" --tip-ratio "$TIP_RATIO"
  --da-phys "$DA_M"
  --target-crack-extension-um "$PROJECTED_STOP_UM"
  --mpz-length-um "$PZ_UM"
  --mpz-n-bins "$PZ_BINS"
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
  --print-every 200
  --save-snapshots 0
  --no-plots
  --out "$RUN_ROOT"
)

{
  echo '#!/usr/bin/env bash'
  printf 'CLEAVAGE_HAZARD_MODE=%q ' "$CLEAVAGE_HAZARD_MODE"
  printf 'CLEAVAGE_HAZARD_SEED=%q ' "$CLEAVAGE_HAZARD_SEED"
  printf 'CLEAVAGE_EVENT_LENGTH_MODE=%q ' "$CLEAVAGE_EVENT_LENGTH_MODE"
  printf 'CLEAVAGE_EVENT_MIN_FACTOR=%q ' "$CLEAVAGE_EVENT_MIN_FACTOR"
  printf 'CLEAVAGE_EVENT_MAX_FACTOR=%q ' "$CLEAVAGE_EVENT_MAX_FACTOR"
  printf 'CLEAVAGE_EVENT_SUBSEGMENT_FRACTION=%q ' "$CLEAVAGE_EVENT_SUBSEGMENT_FRACTION"
  printf 'PERSISTENT_SOURCE_MIN_WIDTH_UM=%q ' "$PERSISTENT_SOURCE_MIN_WIDTH_UM"
  printf '%q ' "${cmd[@]}"
  printf '\n'
} > "$RUN_ROOT/command.sh"
chmod +x "$RUN_ROOT/command.sh"

"$PYTHON_BIN" - "$RUN_ROOT" "$CONFIG" "$STATE_TABLE" "$SEED_FAMILY" "$SEED_FAMILY_SHA256" "$REGISTRY" "$KERNEL_CAPTURE_PARAMETER_OPTION" "$KERNEL_CAPTURE_HAZARD_SEED" "$MAX_ANCHOR_UM" "$PROJECTED_STOP_UM" "$REQUIRED_SEED_PATH_UM" <<'PY'
import json
from pathlib import Path
import sys
(
    run_root,
    config,
    state_table,
    family,
    family_sha,
    registry,
    option,
    seed,
    maximum_anchor,
    projected_stop,
    required_seed_path,
) = sys.argv[1:]
payload = {
    "schema": "v10.2.27_accepted_production_kernel_capture_command_v3",
    "mechanical_configuration": str(Path(config).resolve()),
    "state_table": str(Path(state_table).resolve()),
    "seed_signed_kernel_family": str(Path(family).resolve()),
    "seed_signed_kernel_family_sha256": family_sha,
    "seed_family_audit": str(Path(run_root, "capture_seed_family_audit.json").resolve()),
    "parameter_registry": str(Path(registry).resolve()),
    "parameter_option": option,
    "hazard_seed": int(seed),
    "maximum_anchor_extension_um": float(maximum_anchor),
    "projected_stop_extension_um": float(projected_stop),
    "required_seed_family_path_extension_um": float(required_seed_path),
    "production_entry": "audited v10.2.27 persistent-site paper stack",
    "stochastic_first_passage": True,
    "variable_event_lengths": True,
    "capture_match_policy": "first accepted state at or above each path anchor",
    "front_state_model": "moving_pz",
    "tip_kinetics_mode": "moving_velocity",
    "persistent_site_source": True,
    "physical_front_width": True,
    "active_shielding": True,
    "signed_active_shielding": True,
    "wake_shielding": False,
    "capture_physics_overrides": [],
}
Path(run_root, "accepted_production_capture_command.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY

printf 'Accepted production kernel capture\n'
printf '  option: %s\n' "$KERNEL_CAPTURE_PARAMETER_OPTION"
printf '  temperature: %s K\n' "$V10227_KERNEL_CAPTURE_TEMPERATURE_K"
printf '  seed family SHA256: %s\n' "$SEED_FAMILY_SHA256"
printf '  anchors through: %s um\n' "$MAX_ANCHOR_UM"
printf '  required seed coverage: %s um\n' "$REQUIRED_SEED_PATH_UM"
printf '  capture root: %s\n' "$SNAPSHOT_ROOT"

"${cmd[@]}" 2>&1 | tee "$RUN_ROOT/run.log"

for required in \
  "$SNAPSHOT_ROOT/capture_complete.json" \
  "$SNAPSHOT_ROOT/kernel_capture_manifest.json"; do
  [[ -f "$required" ]] || { echo "ERROR: capture did not create $required" >&2; exit 2; }
done
cp "$RUN_ROOT/command.sh" "$SNAPSHOT_ROOT/accepted_production_capture_command.sh"
cp \
  "$RUN_ROOT/accepted_production_capture_command.json" \
  "$SNAPSHOT_ROOT/accepted_production_capture_command.json"
cp \
  "$RUN_ROOT/capture_seed_family_audit.json" \
  "$SNAPSHOT_ROOT/capture_seed_family_audit.json"
