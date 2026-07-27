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

OUTROOT=${OUTROOT:-runs/v10_2_28_paper_four_class_1000um_theta30_varseed_base3621_v1}
KERNEL_CACHE_ROOT=${KERNEL_CACHE_ROOT:-runs/v10_2_28_kernel_cache}
TARGET_EXT_UM=${TARGET_EXT_UM:-1000}
THETA=${THETA:-30}
TEMPS=${TEMPS:-"300 600 800 900 950 1000 1050 1100 1150 1200 1250 1300"}
OPTIONS=${OPTIONS:-"v913_paper_peak01_0242980_persistent_sites v913_paper_dbtt01_0202500_persistent_sites v913_paper_weakT01_0129902_persistent_sites v913_paper_ceramic01_0077080_persistent_sites"}
MAX_JOBS=${MAX_JOBS:-2}
HAZARD_SEED=${HAZARD_SEED:-3621}
SEED_OPTION_STRIDE=${SEED_OPTION_STRIDE:-1000000}
SEED_TEMPERATURE_STRIDE=${SEED_TEMPERATURE_STRIDE:-1009}
STEPS=${STEPS:-2000000}
SAVE_SNAPSHOTS=${SAVE_SNAPSHOTS:-20}
SNAPSHOT_COLS=${SNAPSHOT_COLS:-5}
PERSISTENT_SOURCE_MIN_WIDTH_UM=${PERSISTENT_SOURCE_MIN_WIDTH_UM:-0}
SKIP_FINISHED=${SKIP_FINISHED:-1}
RESTART_INCOMPLETE=${RESTART_INCOMPLETE:-1}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}

TARGET_EXT_UM="$TARGET_EXT_UM" THETA="$THETA" "$PYTHON_BIN" - <<'PY'
import math
import os

target = float(os.environ["TARGET_EXT_UM"])
theta = float(os.environ["THETA"])
if not math.isclose(target, 1000.0, rel_tol=0.0, abs_tol=1.0e-12):
    raise SystemExit("ERROR: this production launcher is fixed to 1000 um crack extension")
if not math.isfinite(theta):
    raise SystemExit("ERROR: THETA must be finite")
PY

SIGNED_KERNEL_NOMINAL_FORWARD_COS=$(THETA="$THETA" "$PYTHON_BIN" - <<'PY'
import math
import os

theta = math.radians(float(os.environ["THETA"]))
# The direct provider chooses the globally most-forward member of the two
# in-plane {100} traces at theta and theta+90 degrees.
value = max(abs(math.cos(theta)), abs(math.sin(theta)))
if not math.isfinite(value) or value <= 1.0e-12:
    raise SystemExit("ERROR: no globally forward direct-provider cleavage trace")
print(f"{value:.17g}")
PY
)
export SIGNED_KERNEL_EXTENSION_COORDINATE=projected_ligament_equivalent
export SIGNED_KERNEL_NOMINAL_FORWARD_COS

mkdir -p "$OUTROOT"

"$PYTHON_BIN" scripts/install_v10_2_27_four_class_registry.py
"$PYTHON_BIN" scripts/install_v10_2_27_four_class_registry.py --check-only

kernel_resolution_tmp="$OUTROOT/.v10_2_28_kernel_resolution.json.tmp"
kernel_resolution="$OUTROOT/v10_2_28_kernel_resolution.json"

"$PYTHON_BIN" scripts/ensure_v10_2_28_signed_kernel.py \
  --theta-deg "$THETA" \
  --target-extension-um "$TARGET_EXT_UM" \
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
  --json > "$kernel_resolution_tmp"

mv "$kernel_resolution_tmp" "$kernel_resolution"

FAMILY_JSON=$(KERNEL_RESOLUTION="$kernel_resolution" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

resolution_path = Path(os.environ["KERNEL_RESOLUTION"]).resolve()
resolution = json.loads(resolution_path.read_text())
family = Path(resolution["family"]).resolve()
if not family.is_file():
    raise SystemExit(f"ERROR: resolved family is missing: {family}")
payload = json.loads(family.read_text())
required = {
    "kernel_provider_id": "v10.2.28_direct_prescribed_geometry_fem_v1",
    "direct_prescribed_geometry": True,
    "prior_kernel_family_required": False,
    "material_parameter_option_required": False,
    "hazard_seed_required": False,
    "stochastic_trajectory_required": False,
    "production_physics_modified": False,
}
for key, expected in required.items():
    if payload.get(key) != expected:
        raise SystemExit(f"ERROR: direct-family provenance mismatch: {key}")
if float(resolution["maximum_extension_um"]) + 1.0e-9 < float(resolution["required_max_extension_um"]):
    raise SystemExit("ERROR: resolved direct family lacks required 1000 um campaign coverage")
print(family)
PY
)

KERNEL_RESOLUTION="$kernel_resolution" FAMILY_JSON="$FAMILY_JSON" OUTROOT="$OUTROOT" \
OPTIONS="$OPTIONS" TEMPS="$TEMPS" TARGET_EXT_UM="$TARGET_EXT_UM" THETA="$THETA" \
SIGNED_KERNEL_EXTENSION_COORDINATE="$SIGNED_KERNEL_EXTENSION_COORDINATE" \
SIGNED_KERNEL_NOMINAL_FORWARD_COS="$SIGNED_KERNEL_NOMINAL_FORWARD_COS" \
"$PYTHON_BIN" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["OUTROOT"]).resolve()
resolution_path = Path(os.environ["KERNEL_RESOLUTION"]).resolve()
resolution = json.loads(resolution_path.read_text())
family = Path(os.environ["FAMILY_JSON"]).resolve()
family_payload = json.loads(family.read_text())

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

payload = {
    "schema": "v10.2.28_paper_four_class_orientation_1000um_campaign_lock_v2",
    "model_entry": "arrhenius_fracture.sharp_front_v10_2_28_audited",
    "target_quantity": "projected_ligament_extension",
    "target_crack_extension_um": float(os.environ["TARGET_EXT_UM"]),
    "kernel_extension_coordinate": os.environ["SIGNED_KERNEL_EXTENSION_COORDINATE"],
    "kernel_nominal_forward_cosine": float(os.environ["SIGNED_KERNEL_NOMINAL_FORWARD_COS"]),
    "kernel_coordinate_definition": (
        "accumulated_actual_projected_ligament_advance_divided_by_"
        "direct_provider_nominal_forward_cosine"
    ),
    "crystal_theta_deg": float(os.environ["THETA"]),
    "options": os.environ["OPTIONS"].split(),
    "temperatures_K": [float(value) for value in os.environ["TEMPS"].split()],
    "planned_case_count": len(os.environ["OPTIONS"].split()) * len(os.environ["TEMPS"].split()),
    "kernel_resolution": str(resolution_path),
    "kernel_resolution_mode": resolution.get("resolution"),
    "kernel_configuration_fingerprint": resolution["configuration_fingerprint"],
    "kernel_required_max_extension_um": resolution["required_max_extension_um"],
    "kernel_maximum_extension_um": resolution["maximum_extension_um"],
    "kernel_family": str(family),
    "kernel_family_sha256": sha256(family),
    "kernel_physics_fingerprint": resolution["physics_fingerprint"],
    "kernel_provider_id": family_payload["kernel_provider_id"],
    "direct_prescribed_geometry": True,
    "production_physics_modified": False,
    "persistent_sites": True,
    "finite_source_inventory": False,
    "source_refresh_on_crack_advance": False,
    "explicit_recovery": False,
    "maximum_fronts": 1,
}
(root / "v10_2_28_campaign_kernel_lock.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

if [[ "$PREFLIGHT_ONLY" == 1 ]]; then
  echo "PREFLIGHT_COMPLETE: direct kernel locked at $FAMILY_JSON"
  exit 0
fi

source_scheduler="$ROOT/scripts/run_v10_2_27_paper_four_class_30deg_long_rcurves.sh"
source_plotter="$ROOT/scripts/plot_v10_2_27_paper_four_class_rcurves.py"
generated_scheduler_base=$(mktemp "$ROOT/scripts/.v10_2_28_four_class_orientation_scheduler.XXXXXX")
generated_scheduler="${generated_scheduler_base}.sh"
mv "$generated_scheduler_base" "$generated_scheduler"
generated_plotter="$OUTROOT/v10_2_28_generated_plotter.py"
cleanup() {
  rm -f "$generated_scheduler"
}
trap cleanup EXIT

SOURCE_SCHEDULER="$source_scheduler" SOURCE_PLOTTER="$source_plotter" \
GENERATED_SCHEDULER="$generated_scheduler" GENERATED_PLOTTER="$generated_plotter" \
OUTROOT="$OUTROOT" "$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path

source_scheduler = Path(os.environ["SOURCE_SCHEDULER"])
source_plotter = Path(os.environ["SOURCE_PLOTTER"])
generated_scheduler = Path(os.environ["GENERATED_SCHEDULER"])
generated_plotter = Path(os.environ["GENERATED_PLOTTER"])
outroot = Path(os.environ["OUTROOT"]).resolve()

replacements = {
    "arrhenius_fracture.sharp_front_v10_2_27_audited":
        "arrhenius_fracture.sharp_front_v10_2_28_audited",
    "v913_paper_weakT01_0257068_persistent_sites":
        "v913_paper_weakT01_0129902_persistent_sites",
    "v913_zeroD_sobol_0257068": "v913_zeroD_sobol_0129902",
    "v913_paper_ceramic01_0189364_persistent_sites":
        "v913_paper_ceramic01_0077080_persistent_sites",
    "v913_zeroD_sobol_0189364": "v913_zeroD_sobol_0077080",
}

scheduler = source_scheduler.read_text()

strict_header = "set -u\nset -o pipefail"
if strict_header not in scheduler:
    raise SystemExit("ERROR: scheduler strict-mode header changed")
scheduler = scheduler.replace(strict_header, "set -euo pipefail", 1)

subset_validation = '''options = os.environ["OPTIONS"].split()
expected_options = [
    "v913_paper_peak01_0242980_persistent_sites",
    "v913_paper_dbtt01_0202500_persistent_sites",
    "v913_paper_weakT01_0129902_persistent_sites",
    "v913_paper_ceramic01_0077080_persistent_sites",
]
if not options:
    raise SystemExit("ERROR: OPTIONS must contain at least one canonical option")
if len(options) != len(set(options)):
    raise SystemExit("ERROR: OPTIONS contains duplicate entries")
unknown_options = [value for value in options if value not in expected_options]
if unknown_options:
    raise SystemExit(f"ERROR: OPTIONS contains unknown entries: {unknown_options}")
option_indices = [expected_options.index(value) for value in options]
if option_indices != sorted(option_indices):
    raise SystemExit(f"ERROR: OPTIONS subset must retain canonical order: {expected_options}")'''
validation_start_marker = '''options = os.environ["OPTIONS"].split()
expected_options = [
'''
validation_end_marker = '''

registry_path = Path(os.environ["REGISTRY"]).resolve()'''
validation_start = scheduler.find(validation_start_marker)
validation_end = scheduler.find(validation_end_marker, validation_start)
if validation_start < 0 or validation_end < 0:
    raise SystemExit("ERROR: scheduler canonical option validation changed")
original_validation = scheduler[validation_start:validation_end]
required_validation_tokens = (
    "if options != expected_options:",
    "ERROR: OPTIONS must retain canonical order",
)
if any(token not in original_validation for token in required_validation_tokens):
    raise SystemExit("ERROR: scheduler canonical option validation changed")
scheduler = (
    scheduler[:validation_start]
    + subset_validation
    + scheduler[validation_end:]
)

seed_loop = "for option_index, option in enumerate(options):"
if seed_loop not in scheduler:
    raise SystemExit("ERROR: scheduler Python seed loop changed")
scheduler = scheduler.replace(
    seed_loop,
    "for option in options:\n    option_index = expected_options.index(option)",
    1,
)

shell_loop = '''option_index=0
for option in $OPTIONS; do
  temperature_index=0'''
shell_subset_loop = '''for option in $OPTIONS; do
  case "$option" in
    v913_paper_peak01_0242980_persistent_sites) option_index=0 ;;
    v913_paper_dbtt01_0202500_persistent_sites) option_index=1 ;;
    v913_paper_weakT01_0129902_persistent_sites) option_index=2 ;;
    v913_paper_ceramic01_0077080_persistent_sites) option_index=3 ;;
    *) echo "ERROR: unknown option in seed map: $option" >&2; exit 2 ;;
  esac
  temperature_index=0'''
if shell_loop not in scheduler:
    raise SystemExit("ERROR: scheduler shell option loop changed")
scheduler = scheduler.replace(shell_loop, shell_subset_loop, 1)
increment = "  option_index=$((option_index + 1))\n"
if increment not in scheduler:
    raise SystemExit("ERROR: scheduler shell option-index increment changed")
scheduler = scheduler.replace(increment, "", 1)

for old, new in replacements.items():
    if old not in scheduler:
        raise SystemExit(f"ERROR: scheduler source no longer contains expected token: {old}")
    scheduler = scheduler.replace(old, new)

plotter = source_plotter.read_text()
for old, new in replacements.items():
    plotter = plotter.replace(old, new)

plot_token = "scripts/plot_v10_2_27_paper_four_class_rcurves.py"
if plot_token not in scheduler:
    raise SystemExit("ERROR: scheduler plotter token is missing")
scheduler = scheduler.replace(plot_token, str(generated_plotter))

generated_scheduler.write_text(scheduler)
generated_plotter.write_text(plotter)
(outroot / "v10_2_28_generated_scheduler.sh").write_text(scheduler)
PY

chmod +x "$generated_scheduler" "$generated_plotter"

export KERNEL_STRICT_FAMILY_OVERRIDE=1
export SIGNED_KERNEL_FAMILY_JSON="$FAMILY_JSON"

env \
  OUTROOT="$OUTROOT" \
  FAMILY_JSON="$FAMILY_JSON" \
  OPTIONS="$OPTIONS" \
  TEMPS="$TEMPS" \
  MAX_JOBS="$MAX_JOBS" \
  HAZARD_SEED="$HAZARD_SEED" \
  SEED_OPTION_STRIDE="$SEED_OPTION_STRIDE" \
  SEED_TEMPERATURE_STRIDE="$SEED_TEMPERATURE_STRIDE" \
  TARGET_EXT_UM="$TARGET_EXT_UM" \
  STEPS="$STEPS" \
  THETA="$THETA" \
  SAVE_SNAPSHOTS="$SAVE_SNAPSHOTS" \
  SNAPSHOT_COLS="$SNAPSHOT_COLS" \
  PERSISTENT_SOURCE_MIN_WIDTH_UM="$PERSISTENT_SOURCE_MIN_WIDTH_UM" \
  SKIP_FINISHED="$SKIP_FINISHED" \
  RESTART_INCOMPLETE="$RESTART_INCOMPLETE" \
  SIGNED_KERNEL_EXTENSION_COORDINATE="$SIGNED_KERNEL_EXTENSION_COORDINATE" \
  SIGNED_KERNEL_NOMINAL_FORWARD_COS="$SIGNED_KERNEL_NOMINAL_FORWARD_COS" \
  bash "$generated_scheduler"
