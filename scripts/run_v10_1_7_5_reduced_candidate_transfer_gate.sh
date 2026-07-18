#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

PYTHON_BIN=${PYTHON_BIN:-python}
SOURCE=${SOURCE:?set SOURCE to the v9.10.4.9 two_d_candidate_ranking.csv}
CANDIDATES=${CANDIDATES:-"DBTT_A0003408 DBTT_A0000353"}
OUTROOT=${OUTROOT:-runs/v10_1_7_5_reduced_candidate_transfer_gate_v1}
TARGET_EXT_UM=${TARGET_EXT_UM:-5}
HEARTBEAT_SECONDS=${HEARTBEAT_SECONDS:-30}
RESUME_VALIDATED=${RESUME_VALIDATED:-1}

STEPS=${STEPS:-4000}
PRINT_EVERY=${PRINT_EVERY:-100}
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
PACKET_LENGTH_M=${PACKET_LENGTH_M:-2.5e-10}
MOBILE_SHIELD_FRACTION=${MOBILE_SHIELD_FRACTION:-1.0}
KINETIC_MAX_ACTION_SUBSTEP=${KINETIC_MAX_ACTION_SUBSTEP:-0.01}
KINETIC_MAX_TRANSLATION_SUBSTEP_M=${KINETIC_MAX_TRANSLATION_SUBSTEP_M:-5e-8}

ANISOTROPIC_PROBE_RADIUS_M=${ANISOTROPIC_PROBE_RADIUS_M:-1e-5}
ANISOTROPIC_PROBE_HALF_ANGLE_DEG=${ANISOTROPIC_PROBE_HALF_ANGLE_DEG:-25}
ANISOTROPIC_PROBE_DAMAGE_CUTOFF=${ANISOTROPIC_PROBE_DAMAGE_CUTOFF:-0.85}
ANISOTROPIC_PROBE_MIN_ELEMENTS=${ANISOTROPIC_PROBE_MIN_ELEMENTS:-3}
ANISOTROPIC_SCHMID_REFERENCE=${ANISOTROPIC_SCHMID_REFERENCE:-0.5}
ANISOTROPIC_SHARED_FOREST_DENSITY=${ANISOTROPIC_SHARED_FOREST_DENSITY:-1}
ANISOTROPIC_REQUIRE_RELIABLE_PROBE=${ANISOTROPIC_REQUIRE_RELIABLE_PROBE:-1}

if [[ -e "$OUTROOT" ]]; then
  echo "ERROR: OUTROOT already exists: $OUTROOT"
  echo "Use a new versioned OUTROOT, or rerun with the same root only after confirming it belongs to this transfer gate."
  exit 1
fi
mkdir -p "$OUTROOT"

"$PYTHON_BIN" scripts/prepare_v10_1_7_5_candidate_transfer.py \
  --source "$SOURCE" \
  --out "$OUTROOT" \
  --candidates $CANDIDATES

CASE_TABLE="$OUTROOT/transfer_cases.tsv"
RUN_MANIFEST="$OUTROOT/transfer_run_manifest.tsv"
printf 'candidate_id\ttransition_bracket\tendpoint\tT_K\tmode\tstatus\toutdir\n' > "$RUN_MANIFEST"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
report() { printf '[%s] %s\n' "$(stamp)" "$*"; }

common_flags() {
  local material_manifest=$1
  printf '%s\n' \
    --mode 2d --material-manifest "$material_manifest" --temperatures "$CURRENT_T" \
    --bulk-plasticity-mode tip_only --directional-j-mode root_signed \
    --tip-kinetics-mode moving_velocity --tip-source-model continuum \
    --signed-active-shielding \
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
    --print-every "$PRINT_EVERY" --save-snapshots 0 --no-plots
}

validate_case() {
  local outdir=$1
  local expected_mode=$2
  local expected_backstress=$3
  local expected_forest=$4
  "$PYTHON_BIN" - "$outdir" "$expected_mode" "$expected_backstress" "$expected_forest" "$TARGET_EXT_UM" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected_mode = sys.argv[2]
expected_backstress = float(sys.argv[3])
expected_forest_text = sys.argv[4]
target_um = float(sys.argv[5])

summary = root / "summary.json"
audit = root / "kinetic_tip_cell_audit_v101.json"
transfer = root / "v10_1_7_5_transfer_gate.json"
mode_path = root / "v10_1_driver_modes.json"
steps = sorted(root.glob("steps_*K.csv"))
assert summary.is_file(), summary
assert audit.is_file(), audit
assert transfer.is_file(), transfer
assert mode_path.is_file(), mode_path
assert len(steps) == 1, steps

transfer_data = json.loads(transfer.read_text())
assert transfer_data["schema"] == "v10.1.7.5_reduced_candidate_transfer_gate"
assert transfer_data["deterministic_hazard"] is True
assert transfer_data["anisotropic_avalanche_backend"] is False
assert transfer_data["single_front"] is True
assert transfer_data["bulk_plasticity_mode"] == "tip_only"
assert abs(float(transfer_data["backstress_scale"]) - expected_backstress) <= 1e-12
if expected_forest_text:
    assert abs(float(transfer_data["forest_density_floor_override_m2"]) - float(expected_forest_text)) <= 1e-12
else:
    assert transfer_data["forest_density_floor_override_m2"] is None

mode = json.loads(mode_path.read_text())
assert mode["anisotropic_emission_enabled"] is True
assert mode["anisotropic_emission_post_hazard_weighting"] is False
assert mode["anisotropic_use_avalanche_backend"] is False
assert mode["bulk_plasticity_mode"] == "tip_only"

kinetics = json.loads(audit.read_text())
records = kinetics.get("records", [])
assert records, audit
assert any(bool(row.get("fired", False)) for row in records), audit
aniso = [row for row in records if "anisotropic_drive_reliable" in row]
assert aniso, "no anisotropic kinetic records"
assert all(bool(row.get("anisotropic_drive_reliable", False)) for row in aniso)
assert all(row.get("anisotropic_post_hazard_weighting_applied") is False for row in aniso)

lines = [line.strip() for line in steps[0].read_text().splitlines() if line.strip()]
header = [token.strip() for token in lines[0].lstrip("# ").split(",")]
index = header.index("crack_extension_m")
extension = max(float(line.split(",")[index]) for line in lines[1:])
assert extension + 1e-12 >= target_um * 1e-6, (extension, target_um)

if expected_mode == "plasticity_off":
    assert mode["tip_plasticity_enabled"] is False
    assert mode["active_shielding_enabled"] is False
elif expected_mode == "shielding_off":
    assert mode["tip_plasticity_enabled"] is True
    assert mode["active_shielding_enabled"] is False
else:
    assert mode["tip_plasticity_enabled"] is True
PY
}

run_case() {
  local candidate=$1
  local bracket=$2
  local endpoint=$3
  local temperature=$4
  local mode=$5
  local material_manifest=$6
  local tip_plasticity=$7
  local active_shielding=$8
  local backstress=$9
  local forest_floor=${10}

  local temperature_tag=${temperature//./p}
  local outdir="$OUTROOT/cases/$candidate/$mode/T${temperature_tag}_th${THETA}"

  if [[ -e "$outdir" ]]; then
    if [[ "$RESUME_VALIDATED" == 1 ]] && validate_case "$outdir" "$mode" "$backstress" "$forest_floor" >/dev/null 2>&1; then
      report "CASE SKIP validated candidate=$candidate mode=$mode T=${temperature}K"
      printf '%s\t%s\t%s\t%s\t%s\tEXISTING\t%s\n' \
        "$candidate" "$bracket" "$endpoint" "$temperature" "$mode" "$outdir" >> "$RUN_MANIFEST"
      return
    fi
    echo "ERROR: existing case is incomplete or inconsistent: $outdir"
    echo "Do not overwrite it. Use a new versioned OUTROOT."
    exit 1
  fi
  mkdir -p "$outdir"

  CURRENT_T=$temperature
  local -a flags=()
  while IFS= read -r flag; do flags+=("$flag"); done < <(common_flags "$material_manifest")
  if [[ "$tip_plasticity" == 1 ]]; then
    flags+=(--tip-plasticity)
  else
    flags+=(--no-tip-plasticity)
  fi
  if [[ "$active_shielding" == 1 ]]; then
    flags+=(--active-shielding)
  else
    flags+=(--no-active-shielding)
  fi

  report "CASE START candidate=$candidate bracket=$bracket endpoint=$endpoint mode=$mode T=${temperature}K out=$outdir"
  local start
  start=$(date +%s)

  env \
    PYTHONUNBUFFERED=1 \
    CAMPAIGN_BACKSTRESS_SCALE=1 \
    CAMPAIGN_REFRESH_SCALE=1 \
    V10175_BACKSTRESS_SCALE="$backstress" \
    V10175_FOREST_DENSITY_FLOOR_M2="$forest_floor" \
    CLEAVAGE_HAZARD_MODE=deterministic \
    CLEAVAGE_HAZARD_SEED=0 \
    CLEAVAGE_EVENT_LENGTH_MODE=fixed \
    ANISOTROPIC_EMISSION_ENABLED=1 \
    ANISOTROPIC_USE_AVALANCHE_BACKEND=0 \
    ANISOTROPIC_PROBE_RADIUS_M="$ANISOTROPIC_PROBE_RADIUS_M" \
    ANISOTROPIC_PROBE_HALF_ANGLE_DEG="$ANISOTROPIC_PROBE_HALF_ANGLE_DEG" \
    ANISOTROPIC_PROBE_DAMAGE_CUTOFF="$ANISOTROPIC_PROBE_DAMAGE_CUTOFF" \
    ANISOTROPIC_PROBE_MIN_ELEMENTS="$ANISOTROPIC_PROBE_MIN_ELEMENTS" \
    ANISOTROPIC_SCHMID_REFERENCE="$ANISOTROPIC_SCHMID_REFERENCE" \
    ANISOTROPIC_SHARED_FOREST_DENSITY="$ANISOTROPIC_SHARED_FOREST_DENSITY" \
    ANISOTROPIC_REQUIRE_RELIABLE_PROBE="$ANISOTROPIC_REQUIRE_RELIABLE_PROBE" \
    "$PYTHON_BIN" -u -m arrhenius_fracture.sharp_front_v10_1_7_5 \
    "${flags[@]}" --out "$outdir" \
    > "$outdir/case.log" 2>&1 &
  local pid=$!

  while kill -0 "$pid" 2>/dev/null; do
    sleep "$HEARTBEAT_SECONDS"
    if kill -0 "$pid" 2>/dev/null; then
      report "HEARTBEAT candidate=$candidate mode=$mode T=${temperature}K elapsed=$(( $(date +%s) - start ))s"
    fi
  done
  if ! wait "$pid"; then
    report "CASE FAILED candidate=$candidate mode=$mode T=${temperature}K log=$outdir/case.log"
    tail -n 80 "$outdir/case.log" || true
    exit 1
  fi

  validate_case "$outdir" "$mode" "$backstress" "$forest_floor"
  printf '%s\t%s\t%s\t%s\t%s\tCOMPLETE\t%s\n' \
    "$candidate" "$bracket" "$endpoint" "$temperature" "$mode" "$outdir" >> "$RUN_MANIFEST"
  report "CASE COMPLETE candidate=$candidate mode=$mode T=${temperature}K elapsed=$(( $(date +%s) - start ))s"
}

{
  read -r _header
  while IFS=$'\t' read -r candidate bracket endpoint temperature mode material_manifest tip_plasticity active_shielding backstress forest_floor; do
    run_case "$candidate" "$bracket" "$endpoint" "$temperature" "$mode" \
      "$material_manifest" "$tip_plasticity" "$active_shielding" "$backstress" "$forest_floor"
  done
} < "$CASE_TABLE"

"$PYTHON_BIN" scripts/analyze_v10_1_7_5_reduced_candidate_transfer.py \
  --root "$OUTROOT"

report "V10.1.7.5 REDUCED-CANDIDATE TRANSFER GATE COMPLETE root=$OUTROOT"
