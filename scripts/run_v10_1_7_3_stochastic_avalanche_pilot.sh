#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN=${PYTHON_BIN:-python}
CLASS=${CLASS:-DBTT}
TEMP_K=${TEMP_K:-700}
SEEDS=${SEEDS:-"1 2"}
TARGET_EXT_UM=${TARGET_EXT_UM:-200}
STEPS=${STEPS:-6000}
PRINT_EVERY=${PRINT_EVERY:-200}
OUTROOT=${OUTROOT:-runs/v10_1_7_3_stochastic_avalanche_DBTT_700K_200um_v1}
FORCE=${FORCE:-0}

CAMPAIGN_BACKSTRESS_SCALE=${CAMPAIGN_BACKSTRESS_SCALE:-1.0}
CAMPAIGN_REFRESH_SCALE=${CAMPAIGN_REFRESH_SCALE:-1.0}
K_FIRST_MAX_MPA_SQRT_M=${K_FIRST_MAX_MPA_SQRT_M:-200}

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
SAVE_SNAPSHOTS=${SAVE_SNAPSHOTS:-0}
PACKET_LENGTH_M=${PACKET_LENGTH_M:-2.5e-10}
MOBILE_SHIELD_FRACTION=${MOBILE_SHIELD_FRACTION:-1.0}
KINETIC_MAX_ACTION_SUBSTEP=${KINETIC_MAX_ACTION_SUBSTEP:-0.01}
KINETIC_MAX_TRANSLATION_SUBSTEP_M=${KINETIC_MAX_TRANSLATION_SUBSTEP_M:-5e-8}
EVENT_MIN_FACTOR=${EVENT_MIN_FACTOR:-0.2}
EVENT_MAX_FACTOR=${EVENT_MAX_FACTOR:-4.0}
EVENT_SUBSEGMENT_FRACTION=${EVENT_SUBSEGMENT_FRACTION:-0.1}

export CAMPAIGN_BACKSTRESS_SCALE CAMPAIGN_REFRESH_SCALE
mkdir -p "$OUTROOT"
MANIFEST="$OUTROOT/stochastic_avalanche_pilot_manifest.tsv"
printf 'case_type\tmode\tseed\tclass\ttemperature_K\ttarget_ext_um\tstatus\toutdir\n' > "$MANIFEST"

validate_case() {
  local outdir=$1
  local case_type=$2
  local mode=$3
  local seed=$4
  "$PYTHON_BIN" - "$outdir" "$case_type" "$mode" "$seed" "$TARGET_EXT_UM" \
    "$DA_CHECKPOINT_M" "$K_FIRST_MAX_MPA_SQRT_M" "$EVENT_MIN_FACTOR" \
    "$EVENT_MAX_FACTOR" <<'PY'
import json, math, pathlib, sys

root = pathlib.Path(sys.argv[1])
case_type = sys.argv[2]
mode = sys.argv[3]
seed = int(sys.argv[4])
target_um = float(sys.argv[5])
da_m = float(sys.argv[6])
kmax = float(sys.argv[7])
min_factor = float(sys.argv[8])
max_factor = float(sys.argv[9])

summary_path = root / "summary.json"
audit_path = root / "kinetic_tip_cell_audit_v101.json"
mode_path = root / "v10_1_driver_modes.json"
if not (summary_path.is_file() and audit_path.is_file() and mode_path.is_file()):
    raise SystemExit(1)

modes = json.loads(mode_path.read_text())
assert modes.get("noise_added_to_K") is False, modes
assert modes.get("noise_added_to_barriers") is False, modes
assert modes.get("wake_shielding") is False, modes
if case_type == "fixed_original":
    assert modes.get("schema") == "v10.1.7.2_stochastic_hazard_pilot", modes
else:
    assert modes.get("schema") == "v10.1.7.3_stochastic_avalanche_length_pilot", modes
    assert modes.get("cleavage_hazard_mode") == mode, modes
    assert int(modes.get("cleavage_hazard_seed", -1)) == seed, modes
    assert modes.get("geometry_subsegments_re_equilibrated") is False, modes

summary = json.loads(summary_path.read_text())
assert summary and isinstance(summary, list), summary
row = summary[0]
kc = float(row["Kc_first_MPa_sqrt_m"])
assert math.isfinite(kc) and 0.0 < kc <= kmax, row
assert int(row.get("n_advances", 0)) > 0, row

audit = json.loads(audit_path.read_text())
records = audit.get("records", [])
assert records, audit
fired = [r for r in records if bool(r.get("fired", False))]
assert fired, audit
final_extension_um = max(
    float(r.get("kinetic_micro_advance_total_m", r.get("micro_advance_total_m", 0.0)))
    for r in records
) * 1.0e6
assert final_extension_um + 1.0e-6 >= target_um, (final_extension_um, target_um)

if case_type == "fixed_original":
    thresholds = [float(r["hazard_last_completed_threshold"]) for r in fired]
    assert all(abs(x - 1.0) <= 1.0e-14 for x in thresholds), thresholds
else:
    lengths = [float(r["avalanche_event_advance_m"]) for r in fired]
    assert all(math.isfinite(x) and x > 0.0 for x in lengths), lengths
    if case_type == "segmented_deterministic":
        assert all(abs(x - da_m) <= 1.0e-12 * max(da_m, 1.0) for x in lengths), lengths
    else:
        lo = da_m * min_factor * 0.99
        hi = da_m * max_factor * 1.01
        assert all(lo <= x <= hi for x in lengths), (lo, hi, lengths)
        assert any(abs(x - da_m) > 1.0e-3 * da_m for x in lengths), lengths
        assert all(r.get("stochastic_avalanche_length_enabled") is True for r in records)
PY
}

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
    --da-phys "$DA_CHECKPOINT_M" --target-crack-extension-um "$TARGET_EXT_UM" \
    --mpz-length-um "$MPZ_LENGTH_UM" --mpz-n-bins "$MPZ_N_BINS" \
    --wake-length-um "$WAKE_LENGTH_UM" --wake-n-bins "$WAKE_N_BINS" \
    --no-wake-shielding --crack-backend sharp_wake \
    --crystal-aniso --crystal-compete --crystal-theta-deg "$THETA" \
    --crystal-material w --j-decomposition cluster \
    --max-fronts 1 --adaptive-events --adaptive-event-target "$EVENT_TARGET" \
    --print-every "$PRINT_EVERY" --save-snapshots "$SAVE_SNAPSHOTS"
}

run_case() {
  local case_type=$1
  local mode=$2
  local seed=$3
  local entry=$4
  local length_mode=$5
  local outdir=$6
  local status=COMPLETE

  mkdir -p "$outdir"
  echo "========================================================================"
  echo "v10.1.7.3 avalanche pilot: case=$case_type mode=$mode seed=$seed"
  echo "class=$CLASS T=${TEMP_K}K target=${TARGET_EXT_UM}um out=$outdir"
  echo "========================================================================"

  if [[ "$FORCE" != 1 ]] && validate_case "$outdir" "$case_type" "$mode" "$seed" >/dev/null 2>&1; then
    echo "SKIP validated complete case: $outdir"
    status=EXISTING
  else
    rm -f "$outdir/summary.json" "$outdir/kinetic_tip_cell_audit_v101.json"
    mapfile -t FLAGS < <(common_flags)
    if ! env \
      CLEAVAGE_HAZARD_MODE="$mode" \
      CLEAVAGE_HAZARD_SEED="$seed" \
      CLEAVAGE_EVENT_LENGTH_MODE="$length_mode" \
      CLEAVAGE_EVENT_MIN_FACTOR="$EVENT_MIN_FACTOR" \
      CLEAVAGE_EVENT_MAX_FACTOR="$EVENT_MAX_FACTOR" \
      CLEAVAGE_EVENT_SUBSEGMENT_FRACTION="$EVENT_SUBSEGMENT_FRACTION" \
      "$PYTHON_BIN" -m "$entry" "${FLAGS[@]}" --out "$outdir"; then
      status=FAILED
    elif ! validate_case "$outdir" "$case_type" "$mode" "$seed"; then
      status=FAILED
    fi
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$case_type" "$mode" "$seed" "$CLASS" "$TEMP_K" "$TARGET_EXT_UM" "$status" "$outdir" \
    >> "$MANIFEST"
  if [[ "$status" == FAILED ]]; then
    echo "ERROR: avalanche pilot case failed validation: $case_type seed=$seed" >&2
    exit 1
  fi
}

run_case fixed_original deterministic 0 \
  arrhenius_fracture.sharp_front_v10_1_7_2 fixed \
  "$OUTROOT/fixed_original/T${TEMP_K}_th${THETA}"

run_case segmented_deterministic deterministic 0 \
  arrhenius_fracture.sharp_front_v10_1_7_3 fixed \
  "$OUTROOT/segmented_deterministic/T${TEMP_K}_th${THETA}"

for SEED in $SEEDS; do
  run_case stochastic_avalanche exponential "$SEED" \
    arrhenius_fracture.sharp_front_v10_1_7_3 threshold_scaled \
    "$OUTROOT/stochastic_avalanche/seed_${SEED}/T${TEMP_K}_th${THETA}"
done

"$PYTHON_BIN" scripts/analyze_v10_1_7_3_stochastic_avalanche_pilot.py \
  --root "$OUTROOT" --class "$CLASS" --temperature "$TEMP_K" \
  --seeds $SEEDS --theta "$THETA" --base-checkpoint-um "$("$PYTHON_BIN" - <<PY
print(float("$DA_CHECKPOINT_M") * 1.0e6)
PY
)"

echo "wrote $MANIFEST"
