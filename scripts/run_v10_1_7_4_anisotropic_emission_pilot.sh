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
SEEDS=${SEEDS:-"1 2"}
TARGET_EXT_UM=${TARGET_EXT_UM:-100}
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
THETA=${THETA:-45}
EVENT_TARGET=${EVENT_TARGET:-0.05}
PACKET_LENGTH_M=${PACKET_LENGTH_M:-2.5e-10}
MOBILE_SHIELD_FRACTION=${MOBILE_SHIELD_FRACTION:-1.0}
KINETIC_MAX_ACTION_SUBSTEP=${KINETIC_MAX_ACTION_SUBSTEP:-0.01}
KINETIC_MAX_TRANSLATION_SUBSTEP_M=${KINETIC_MAX_TRANSLATION_SUBSTEP_M:-5e-8}
EVENT_MIN_FACTOR=${EVENT_MIN_FACTOR:-0.5}
EVENT_MAX_FACTOR=${EVENT_MAX_FACTOR:-4.0}
EVENT_SUBSEGMENT_FRACTION=${EVENT_SUBSEGMENT_FRACTION:-0.1}

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
  "$PYTHON_BIN" - <<PY
import hashlib, json
payload = {
    "git": "$GIT_HEAD",
    "class": "$CLASS",
    "T": "$TEMP_K",
    "seeds": "$SEEDS",
    "target_um": "$TARGET_EXT_UM",
    "theta": "$THETA",
    "probe_m": "$ANISOTROPIC_PROBE_RADIUS_M",
    "probe_angle": "$ANISOTROPIC_PROBE_HALF_ANGLE_DEG",
    "damage_cutoff": "$ANISOTROPIC_PROBE_DAMAGE_CUTOFF",
    "schmid_reference": "$ANISOTROPIC_SCHMID_REFERENCE",
}
print(hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12])
PY
)
OUTROOT=${OUTROOT:-runs/v10_1_7_4_anisotropic_emission_${CLASS}_${TEMP_K}K_${TARGET_EXT_UM}um_${GIT_SHORT}_${CONFIG_SIGNATURE}}
mkdir -p "$OUTROOT"
MANIFEST="$OUTROOT/anisotropic_emission_pilot_manifest.tsv"
printf 'case_type\tmode\tseed\tanisotropic\tavalanche_backend\tstatus\toutdir\n' > "$MANIFEST"

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
  local anisotropic=$2
  local avalanche_backend=$3
  "$PYTHON_BIN" - "$outdir" "$anisotropic" "$avalanche_backend" "$TARGET_EXT_UM" <<'PY'
import json, math, pathlib, sys
root = pathlib.Path(sys.argv[1])
anisotropic = bool(int(sys.argv[2]))
avalanche_backend = bool(int(sys.argv[3]))
target_um = float(sys.argv[4])

summary_path = root / "summary.json"
audit_path = root / "kinetic_tip_cell_audit_v101.json"
step_paths = sorted(root.glob("steps_*K.csv"))
assert summary_path.is_file(), summary_path
assert audit_path.is_file(), audit_path
assert len(step_paths) == 1, step_paths

audit = json.loads(audit_path.read_text())
records = audit.get("records", [])
assert records, audit_path
assert any(bool(row.get("fired", False)) for row in records), audit_path

lines = [line.strip() for line in step_paths[0].read_text().splitlines() if line.strip()]
header = [token.strip() for token in lines[0].lstrip("# ").split(",")]
idx = header.index("crack_extension_m")
extension = max(float(line.split(",")[idx]) for line in lines[1:])
assert extension + 1e-12 >= target_um * 1e-6, (extension, target_um)

if anisotropic:
    mode = json.loads((root / "v10_1_driver_modes.json").read_text())
    assert mode["schema"] == "v10.1.7.4_anisotropic_emission_pilot", mode
    assert mode["anisotropic_emission_enabled"] is True, mode
    assert mode["anisotropic_emission_post_hazard_weighting"] is False, mode
    assert bool(mode["anisotropic_use_avalanche_backend"]) == avalanche_backend, mode
    aniso = json.loads((root / "anisotropic_emission_audit_v10174.json").read_text())
    cfg = aniso.get("anisotropic_emission", {})
    assert cfg.get("post_hazard_directional_weighting") is False, cfg
    aniso_records = [
        row for row in records
        if "anisotropic_drive_reliable" in row
    ]
    assert aniso_records, "no anisotropic audit records"
    assert all(row.get("anisotropic_post_hazard_weighting_applied") is False
               for row in aniso_records), aniso_records[-1]
    assert all(bool(row.get("anisotropic_drive_reliable", False))
               for row in aniso_records), aniso_records[-1]

if avalanche_backend:
    geometry = root / "stochastic_avalanche_geometry_events.json"
    assert geometry.is_file(), geometry
PY
}

run_case() {
  local case_type=$1
  local mode=$2
  local seed=$3
  local entry=$4
  local length_mode=$5
  local anisotropic=$6
  local avalanche_backend=$7
  local outdir=$8

  if [[ "$FORCE" == 1 ]]; then
    rm -rf "$outdir"
  elif validate_case "$outdir" "$anisotropic" "$avalanche_backend" >/dev/null 2>&1; then
    report "CASE SKIP validated case=$case_type seed=$seed"
    printf '%s\t%s\t%s\t%s\t%s\tEXISTING\t%s\n' \
      "$case_type" "$mode" "$seed" "$anisotropic" "$avalanche_backend" "$outdir" >> "$MANIFEST"
    return
  fi
  mkdir -p "$outdir"

  local -a flags=()
  while IFS= read -r flag; do flags+=("$flag"); done < <(common_flags)

  report "CASE START case=$case_type mode=$mode seed=$seed out=$outdir"
  env \
    PYTHONUNBUFFERED=1 \
    CLEAVAGE_HAZARD_MODE="$mode" \
    CLEAVAGE_HAZARD_SEED="$seed" \
    CLEAVAGE_EVENT_LENGTH_MODE="$length_mode" \
    CLEAVAGE_EVENT_MIN_FACTOR="$EVENT_MIN_FACTOR" \
    CLEAVAGE_EVENT_MAX_FACTOR="$EVENT_MAX_FACTOR" \
    CLEAVAGE_EVENT_SUBSEGMENT_FRACTION="$EVENT_SUBSEGMENT_FRACTION" \
    ANISOTROPIC_EMISSION_ENABLED="$anisotropic" \
    ANISOTROPIC_USE_AVALANCHE_BACKEND="$avalanche_backend" \
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
      report "HEARTBEAT case=$case_type seed=$seed elapsed=$(( $(date +%s) - start ))s"
    fi
  done
  wait "$pid"
  validate_case "$outdir" "$anisotropic" "$avalanche_backend"
  printf '%s\t%s\t%s\t%s\t%s\tCOMPLETE\t%s\n' \
    "$case_type" "$mode" "$seed" "$anisotropic" "$avalanche_backend" "$outdir" >> "$MANIFEST"
  report "CASE COMPLETE case=$case_type seed=$seed elapsed=$(( $(date +%s) - start ))s"
}

TAG="T${TEMP_K}_th${THETA}"

# Scalar constitutive reference.
run_case scalar_reference deterministic 0 \
  arrhenius_fracture.sharp_front_v10_1_7_3 fixed 0 1 \
  "$OUTROOT/scalar_reference/$TAG"

# Anisotropic deterministic control without the avalanche geometry wrapper.
run_case fixed_original deterministic 0 \
  arrhenius_fracture.sharp_front_v10_1_7_4 fixed 1 0 \
  "$OUTROOT/fixed_original/$TAG"

# Same anisotropic physics through the corrected avalanche wrapper.
run_case segmented_deterministic deterministic 0 \
  arrhenius_fracture.sharp_front_v10_1_7_4 fixed 1 1 \
  "$OUTROOT/segmented_deterministic/$TAG"

for seed in $SEEDS; do
  run_case stochastic_avalanche exponential "$seed" \
    arrhenius_fracture.sharp_front_v10_1_7_4 threshold_scaled 1 1 \
    "$OUTROOT/stochastic_avalanche/seed_${seed}/$TAG"
done

"$PYTHON_BIN" scripts/analyze_v10_1_7_3_stochastic_avalanche_pilot.py \
  --root "$OUTROOT" --class "$CLASS" --temperature "$TEMP_K" \
  --seeds $SEEDS --theta "$THETA" \
  --base-checkpoint-um "$("$PYTHON_BIN" - <<PY
print(float("$DA_CHECKPOINT_M") * 1e6)
PY
)"

"$PYTHON_BIN" scripts/analyze_v10_1_7_4_anisotropic_emission_pilot.py \
  --root "$OUTROOT" --temperature "$TEMP_K" --theta "$THETA" --seeds $SEEDS

report "ANISOTROPIC EMISSION PILOT COMPLETE root=$OUTROOT"
