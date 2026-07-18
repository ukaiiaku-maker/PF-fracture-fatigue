#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

PYTHON_BIN=${PYTHON_BIN:-python}
CLASS=${CLASS:-DBTT}
TEMPS=${TEMPS:-"500 700 900"}
THETAS=${THETAS:-"15 30 45"}
TARGET_EXT_UM=${TARGET_EXT_UM:-75}
STEPS=${STEPS:-4000}
PRINT_EVERY=${PRINT_EVERY:-100}
HEARTBEAT_SECONDS=${HEARTBEAT_SECONDS:-30}
FORCE=${FORCE:-1}

NX=${NX:-48}
NY=${NY:-96}
TIP_H_FINE=${TIP_H_FINE:-5e-7}
TIP_RATIO=${TIP_RATIO:-1.2}
DU=${DU:-2e-7}
DT=${DT:-8.4}
N_STAGGER=${N_STAGGER:-2}
DA_CHECKPOINT_M=${DA_CHECKPOINT_M:-5e-6}
MPZ_LENGTH_UM=${MPZ_LENGTH_UM:-100}
MPZ_N_BINS=${MPZ_N_BINS:-200}
WAKE_LENGTH_UM=${WAKE_LENGTH_UM:-100}
WAKE_N_BINS=${WAKE_N_BINS:-0}
EVENT_TARGET=${EVENT_TARGET:-0.05}
PACKET_LENGTH_M=${PACKET_LENGTH_M:-2.5e-10}
MOBILE_SHIELD_FRACTION=${MOBILE_SHIELD_FRACTION:-1.0}
KINETIC_MAX_ACTION_SUBSTEP=${KINETIC_MAX_ACTION_SUBSTEP:-0.01}
KINETIC_MAX_TRANSLATION_SUBSTEP_M=${KINETIC_MAX_TRANSLATION_SUBSTEP_M:-5e-8}

CAMPAIGN_BACKSTRESS_SCALE=${CAMPAIGN_BACKSTRESS_SCALE:-1.0}
CAMPAIGN_REFRESH_SCALE=${CAMPAIGN_REFRESH_SCALE:-1.0}
ANISOTROPIC_PROBE_RADIUS_M=${ANISOTROPIC_PROBE_RADIUS_M:-1e-5}
ANISOTROPIC_PROBE_HALF_ANGLE_DEG=${ANISOTROPIC_PROBE_HALF_ANGLE_DEG:-25}
ANISOTROPIC_PROBE_DAMAGE_CUTOFF=${ANISOTROPIC_PROBE_DAMAGE_CUTOFF:-0.85}
ANISOTROPIC_PROBE_MIN_ELEMENTS=${ANISOTROPIC_PROBE_MIN_ELEMENTS:-3}
ANISOTROPIC_SCHMID_REFERENCE=${ANISOTROPIC_SCHMID_REFERENCE:-0.5}
ANISOTROPIC_SHARED_FOREST_DENSITY=${ANISOTROPIC_SHARED_FOREST_DENSITY:-1}
ANISOTROPIC_REQUIRE_RELIABLE_PROBE=${ANISOTROPIC_REQUIRE_RELIABLE_PROBE:-1}

export CAMPAIGN_BACKSTRESS_SCALE CAMPAIGN_REFRESH_SCALE

GIT_HEAD=$(git rev-parse HEAD)
GIT_SHORT=$(git rev-parse --short HEAD)
CONFIG_SIGNATURE=$(
  SWEEP_GIT_HEAD="$GIT_HEAD" \
  SWEEP_CLASS="$CLASS" \
  SWEEP_TEMPS="$TEMPS" \
  SWEEP_THETAS="$THETAS" \
  SWEEP_TARGET="$TARGET_EXT_UM" \
  SWEEP_PROBE="$ANISOTROPIC_PROBE_RADIUS_M" \
  SWEEP_ANGLE="$ANISOTROPIC_PROBE_HALF_ANGLE_DEG" \
  SWEEP_DAMAGE="$ANISOTROPIC_PROBE_DAMAGE_CUTOFF" \
  SWEEP_SCHMID="$ANISOTROPIC_SCHMID_REFERENCE" \
  "$PYTHON_BIN" - <<'PY'
import hashlib, json, os
payload = {key: os.environ[key] for key in sorted(k for k in os.environ if k.startswith("SWEEP_"))}
print(hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12])
PY
)

OUTROOT=${OUTROOT:-runs/v10_1_7_4_orientation_temperature_${CLASS}_${TARGET_EXT_UM}um_${GIT_SHORT}_${CONFIG_SIGNATURE}}
mkdir -p "$OUTROOT"
MANIFEST="$OUTROOT/orientation_temperature_manifest.tsv"
printf 'model\ttemperature_K\ttheta_deg\tstatus\toutdir\n' > "$MANIFEST"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
report() { printf '[%s] %s\n' "$(stamp)" "$*"; }

common_flags() {
  local temp=$1
  local theta=$2
  printf '%s\n' \
    --mode 2d --material-class "$CLASS" --temperatures "$temp" \
    --bulk-plasticity-mode tip_only --directional-j-mode root_signed \
    --tip-kinetics-mode moving_velocity --tip-source-model continuum \
    --tip-plasticity --active-shielding --signed-active-shielding \
    --mobile-shield-fraction "$MOBILE_SHIELD_FRACTION" \
    --kinetic-packet-length-m "$PACKET_LENGTH_M" \
    --kinetic-max-action-substep "$KINETIC_MAX_ACTION_SUBSTEP" \
    --kinetic-max-translation-substep-m "$KINETIC_MAX_TRANSLATION_SUBSTEP_M" \
    --steps "$STEPS" --nx "$NX" --ny "$NY" \
    --dU "$DU" --dt "$DT" --n-stagger "$N_STAGGER" \
    --tip-h-fine "$TIP_H_FINE" --tip-ratio "$TIP_RATIO" \
    --da-phys "$DA_CHECKPOINT_M" \
    --target-crack-extension-um "$TARGET_EXT_UM" \
    --mpz-length-um "$MPZ_LENGTH_UM" --mpz-n-bins "$MPZ_N_BINS" \
    --wake-length-um "$WAKE_LENGTH_UM" --wake-n-bins "$WAKE_N_BINS" \
    --no-wake-shielding --crack-backend sharp_wake \
    --crystal-aniso --crystal-compete --crystal-theta-deg "$theta" \
    --crystal-material w --j-decomposition cluster \
    --max-fronts 1 --adaptive-events \
    --adaptive-event-target "$EVENT_TARGET" \
    --print-every "$PRINT_EVERY" --save-snapshots 0 --no-plots
}

validate_case() {
  local outdir=$1
  local model=$2
  local target_um=$3
  "$PYTHON_BIN" - "$outdir" "$model" "$target_um" <<'PY'
import json, math, pathlib, sys
root = pathlib.Path(sys.argv[1])
model = sys.argv[2]
target_um = float(sys.argv[3])
summary_path = root / "summary.json"
audit_path = root / "kinetic_tip_cell_audit_v101.json"
step_paths = sorted(root.glob("steps_*K.csv"))
assert summary_path.is_file(), summary_path
assert audit_path.is_file(), audit_path
assert len(step_paths) == 1, step_paths
summary = json.loads(summary_path.read_text())
assert len(summary) == 1, summary
assert math.isfinite(float(summary[0]["Kc_first_MPa_sqrt_m"])), summary[0]
audit = json.loads(audit_path.read_text())
records = audit.get("records", [])
assert records and any(bool(row.get("fired", False)) for row in records), audit_path
lines = [line.strip() for line in step_paths[0].read_text().splitlines() if line.strip()]
header = [token.strip() for token in lines[0].lstrip("# ").split(",")]
idx = header.index("crack_extension_m")
extension = max(float(line.split(",")[idx]) for line in lines[1:])
assert extension + 1e-12 >= target_um * 1e-6, (extension, target_um)
if model == "anisotropic":
    aniso_path = root / "anisotropic_emission_audit_v10174.json"
    assert aniso_path.is_file(), aniso_path
    mode = json.loads((root / "v10_1_driver_modes.json").read_text())
    assert mode.get("schema") == "v10.1.7.4_anisotropic_emission_pilot", mode
    assert mode.get("anisotropic_emission_enabled") is True, mode
    assert mode.get("anisotropic_use_avalanche_backend") is False, mode
    assert mode.get("anisotropic_emission_post_hazard_weighting") is False, mode
    rows = [row for row in records if "anisotropic_drive_reliable" in row]
    assert rows, "no anisotropic records"
    assert all(bool(row.get("anisotropic_drive_reliable", False)) for row in rows), rows[-1]
    assert all(row.get("anisotropic_post_hazard_weighting_applied") is False for row in rows), rows[-1]
    assert any(max(row.get("anisotropic_drive_factors", [0.0])) > 0.0 for row in rows), rows[-1]
PY
}

run_case() {
  local model=$1
  local temp=$2
  local theta=$3
  local entry=$4
  local anisotropic=$5
  local outdir=$6

  if [[ "$FORCE" == 1 ]]; then
    rm -rf "$outdir"
  elif validate_case "$outdir" "$model" "$TARGET_EXT_UM" >/dev/null 2>&1; then
    report "CASE SKIP model=$model T=${temp}K theta=${theta} status=EXISTING"
    printf '%s\t%s\t%s\tEXISTING\t%s\n' "$model" "$temp" "$theta" "$outdir" >> "$MANIFEST"
    return
  fi
  mkdir -p "$outdir"

  local -a flags=()
  while IFS= read -r flag; do flags+=("$flag"); done < <(common_flags "$temp" "$theta")

  report "CASE START model=$model T=${temp}K theta=${theta} out=$outdir"
  env \
    PYTHONUNBUFFERED=1 \
    CLEAVAGE_HAZARD_MODE=deterministic \
    CLEAVAGE_HAZARD_SEED=0 \
    CLEAVAGE_EVENT_LENGTH_MODE=fixed \
    ANISOTROPIC_EMISSION_ENABLED="$anisotropic" \
    ANISOTROPIC_USE_AVALANCHE_BACKEND=0 \
    ANISOTROPIC_PROBE_RADIUS_M="$ANISOTROPIC_PROBE_RADIUS_M" \
    ANISOTROPIC_PROBE_HALF_ANGLE_DEG="$ANISOTROPIC_PROBE_HALF_ANGLE_DEG" \
    ANISOTROPIC_PROBE_DAMAGE_CUTOFF="$ANISOTROPIC_PROBE_DAMAGE_CUTOFF" \
    ANISOTROPIC_PROBE_MIN_ELEMENTS="$ANISOTROPIC_PROBE_MIN_ELEMENTS" \
    ANISOTROPIC_SCHMID_REFERENCE="$ANISOTROPIC_SCHMID_REFERENCE" \
    ANISOTROPIC_SHARED_FOREST_DENSITY="$ANISOTROPIC_SHARED_FOREST_DENSITY" \
    ANISOTROPIC_REQUIRE_RELIABLE_PROBE="$ANISOTROPIC_REQUIRE_RELIABLE_PROBE" \
    "$PYTHON_BIN" -u -m "$entry" "${flags[@]}" --out "$outdir" &
  local pid=$!
  local start=$(date +%s)
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$HEARTBEAT_SECONDS"
    if kill -0 "$pid" 2>/dev/null; then
      report "HEARTBEAT model=$model T=${temp}K theta=${theta} elapsed=$(( $(date +%s) - start ))s"
    fi
  done
  wait "$pid"
  validate_case "$outdir" "$model" "$TARGET_EXT_UM"
  printf '%s\t%s\t%s\tCOMPLETE\t%s\n' "$model" "$temp" "$theta" "$outdir" >> "$MANIFEST"
  report "CASE COMPLETE model=$model T=${temp}K theta=${theta} elapsed=$(( $(date +%s) - start ))s"
}

for theta in $THETAS; do
  for temp in $TEMPS; do
    tag="T${temp}_th${theta}"
    # Paired scalar-emission control at the identical temperature, orientation,
    # elasticity, cleavage competition, mesh, loading, and crack-path settings.
    run_case scalar "$temp" "$theta" \
      arrhenius_fracture.sharp_front_v10_1_7_3 0 \
      "$OUTROOT/scalar/$tag"

    run_case anisotropic "$temp" "$theta" \
      arrhenius_fracture.sharp_front_v10_1_7_4 1 \
      "$OUTROOT/anisotropic/$tag"
  done
done

"$PYTHON_BIN" scripts/analyze_v10_1_7_4_orientation_temperature_sweep.py \
  --root "$OUTROOT" --temperatures $TEMPS --thetas $THETAS \
  --target-extension-um "$TARGET_EXT_UM"

report "ORIENTATION-TEMPERATURE SWEEP COMPLETE root=$OUTROOT"
