#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

PYTHON_BIN=${PYTHON_BIN:-python}
CLASS=${CLASS:-DBTT}
TEMP_K=${TEMP_K:-700}
THETA=${THETA:-45}
TARGET_EXT_UM=${TARGET_EXT_UM:-500}
STEPS=${STEPS:-15000}
PRINT_EVERY=${PRINT_EVERY:-100}
HEARTBEAT_SECONDS=${HEARTBEAT_SECONDS:-60}
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
  COMPARE_GIT_HEAD="$GIT_HEAD" \
  COMPARE_CLASS="$CLASS" \
  COMPARE_TEMP="$TEMP_K" \
  COMPARE_THETA="$THETA" \
  COMPARE_TARGET="$TARGET_EXT_UM" \
  COMPARE_PROBE="$ANISOTROPIC_PROBE_RADIUS_M" \
  COMPARE_ANGLE="$ANISOTROPIC_PROBE_HALF_ANGLE_DEG" \
  COMPARE_DAMAGE="$ANISOTROPIC_PROBE_DAMAGE_CUTOFF" \
  COMPARE_SCHMID="$ANISOTROPIC_SCHMID_REFERENCE" \
  "$PYTHON_BIN" - <<'PY'
import hashlib, json, os
payload = {key: os.environ[key] for key in sorted(k for k in os.environ if k.startswith("COMPARE_"))}
print(hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12])
PY
)

OUTROOT=${OUTROOT:-runs/v10_1_7_5_transport_comparison_${CLASS}_${TEMP_K}K_th${THETA}_${TARGET_EXT_UM}um_${GIT_SHORT}_${CONFIG_SIGNATURE}}
mkdir -p "$OUTROOT"
MANIFEST="$OUTROOT/transport_comparison_manifest.tsv"
printf 'case\ttransport_mode\tstatus\toutdir\n' > "$MANIFEST"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
report() { printf '[%s] %s\n' "$(stamp)" "$*"; }

common_flags() {
  printf '%s\n' \
    --mode 2d --material-class "$CLASS" --temperatures "$TEMP_K" \
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
    --crystal-aniso --crystal-compete --crystal-theta-deg "$THETA" \
    --crystal-material w --j-decomposition cluster \
    --max-fronts 1 --adaptive-events \
    --adaptive-event-target "$EVENT_TARGET" \
    --print-every "$PRINT_EVERY" --save-snapshots 0 --no-plots
}

validate_case() {
  local outdir=$1
  local expected_mode=$2
  "$PYTHON_BIN" - "$outdir" "$expected_mode" "$TARGET_EXT_UM" <<'PY'
import json, math, pathlib, sys
root = pathlib.Path(sys.argv[1])
expected_mode = sys.argv[2]
target_um = float(sys.argv[3])
summary = json.loads((root / "summary.json").read_text())
assert len(summary) == 1, summary
assert math.isfinite(float(summary[0]["Kc_first_MPa_sqrt_m"])), summary[0]
audit = json.loads((root / "kinetic_tip_cell_audit_v101.json").read_text())
records = audit.get("records", [])
assert records and any(bool(row.get("fired", False)) for row in records), root
step_paths = sorted(root.glob("steps_*K.csv"))
assert len(step_paths) == 1, step_paths
lines = [line.strip() for line in step_paths[0].read_text().splitlines() if line.strip()]
header = [token.strip() for token in lines[0].lstrip("# ").split(",")]
idx = header.index("crack_extension_m")
extension = max(float(line.split(",")[idx]) for line in lines[1:]) * 1e6
assert extension + 1e-6 >= target_um, (extension, target_um)
if expected_mode != "scalar_reference":
    transport = json.loads((root / "v10_1_7_5_transport_mode.json").read_text())
    assert transport["transport_mode"] == expected_mode, transport
    rows = [row for row in records if "anisotropic_drive_reliable" in row]
    assert rows, "no anisotropic records"
    assert all(bool(row.get("anisotropic_drive_reliable", False)) for row in rows), rows[-1]
    assert all(row.get("anisotropic_post_hazard_weighting_applied") is False for row in rows), rows[-1]
    assert all(row.get("anisotropic_transport_mode") == expected_mode for row in rows), rows[-1]
PY
}

run_case() {
  local case_name=$1
  local transport_mode=$2
  local entry=$3
  local anisotropic=$4
  local outdir=$5

  if [[ "$FORCE" == 1 ]]; then
    rm -rf "$outdir"
  elif validate_case "$outdir" "$transport_mode" >/dev/null 2>&1; then
    report "CASE SKIP case=$case_name mode=$transport_mode status=EXISTING"
    printf '%s\t%s\tEXISTING\t%s\n' "$case_name" "$transport_mode" "$outdir" >> "$MANIFEST"
    return
  fi
  mkdir -p "$outdir"

  local -a flags=()
  while IFS= read -r flag; do flags+=("$flag"); done < <(common_flags)

  report "CASE START case=$case_name mode=$transport_mode out=$outdir"
  env \
    PYTHONUNBUFFERED=1 \
    CLEAVAGE_HAZARD_MODE=deterministic \
    CLEAVAGE_HAZARD_SEED=0 \
    CLEAVAGE_EVENT_LENGTH_MODE=fixed \
    ANISOTROPIC_EMISSION_ENABLED="$anisotropic" \
    ANISOTROPIC_USE_AVALANCHE_BACKEND=0 \
    ANISOTROPIC_TRANSPORT_MODE="$transport_mode" \
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
      report "HEARTBEAT case=$case_name mode=$transport_mode elapsed=$(( $(date +%s) - start ))s"
    fi
  done
  wait "$pid"
  validate_case "$outdir" "$transport_mode"
  printf '%s\t%s\tCOMPLETE\t%s\n' "$case_name" "$transport_mode" "$outdir" >> "$MANIFEST"
  report "CASE COMPLETE case=$case_name mode=$transport_mode elapsed=$(( $(date +%s) - start ))s"
}

TAG="T${TEMP_K}_th${THETA}"

# Frozen campaign-calibrated reference. This is not part of the isolated
# transport pair, but shows whether either new path departs from the repaired
# production R-curve.
run_case scalar_reference scalar_reference \
  arrhenius_fracture.sharp_front_v10_1_7_3 0 \
  "$OUTROOT/scalar_reference/$TAG"

# Same anisotropic emission, campaign finite source budget, geometry refresh,
# shielding cap, and FEM tensor drive. Only the active-zone transport differs.
run_case anisotropic_validated_transport validated_scalar \
  arrhenius_fracture.sharp_front_v10_1_7_5 1 \
  "$OUTROOT/anisotropic_validated_transport/$TAG"

run_case anisotropic_channel_transport channel_resolved \
  arrhenius_fracture.sharp_front_v10_1_7_5 1 \
  "$OUTROOT/anisotropic_channel_transport/$TAG"

"$PYTHON_BIN" scripts/analyze_v10_1_7_5_transport_comparison.py \
  --root "$OUTROOT" --temperature "$TEMP_K" --theta "$THETA" \
  --target-extension-um "$TARGET_EXT_UM"

report "TRANSPORT COMPARISON COMPLETE root=$OUTROOT"
