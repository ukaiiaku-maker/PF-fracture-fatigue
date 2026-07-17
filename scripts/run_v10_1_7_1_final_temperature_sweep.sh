#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN=${PYTHON_BIN:-python}
CLASSES=${CLASSES:-"ceramic weakT DBTT"}
TEMPS=${TEMPS:-"300 400 500 600 700 800 900 1000 1100"}
MODES=${MODES:-"full"}
TARGET_EXT_UM=${TARGET_EXT_UM:-500}
STEPS=${STEPS:-12000}
PRINT_EVERY=${PRINT_EVERY:-200}
OUTROOT=${OUTROOT:-runs/v10_1_7_1_final_production_three_class_500um_v1}
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
SNAPSHOT_BY_EXT_UM=${SNAPSHOT_BY_EXT_UM:-25}
PACKET_LENGTH_M=${PACKET_LENGTH_M:-2.5e-10}
MOBILE_SHIELD_FRACTION=${MOBILE_SHIELD_FRACTION:-1.0}
KINETIC_MAX_ACTION_SUBSTEP=${KINETIC_MAX_ACTION_SUBSTEP:-0.01}
KINETIC_MAX_TRANSLATION_SUBSTEP_M=${KINETIC_MAX_TRANSLATION_SUBSTEP_M:-5e-8}

export CAMPAIGN_BACKSTRESS_SCALE CAMPAIGN_REFRESH_SCALE
mkdir -p "$OUTROOT"
MANIFEST="$OUTROOT/final_production_manifest.tsv"
printf 'mode\tclass\ttemperature_K\ttarget_ext_um\tstatus\toutdir\n' > "$MANIFEST"

validate_case() {
  local outdir=$1
  local mode=$2
  "$PYTHON_BIN" - "$outdir" "$mode" "$TARGET_EXT_UM" "$DA_CHECKPOINT_M" \
    "$K_FIRST_MAX_MPA_SQRT_M" "$CAMPAIGN_BACKSTRESS_SCALE" \
    "$CAMPAIGN_REFRESH_SCALE" <<'PY'
import json, math, pathlib, sys

root = pathlib.Path(sys.argv[1])
run_mode = sys.argv[2]
target_um = float(sys.argv[3])
da_m = float(sys.argv[4])
kmax = float(sys.argv[5])
back_scale = float(sys.argv[6])
refresh_scale = float(sys.argv[7])

summary_path = root / "summary.json"
audit_path = root / "kinetic_tip_cell_audit_v101.json"
mode_path = root / "v10_1_driver_modes.json"
if not (summary_path.is_file() and audit_path.is_file() and mode_path.is_file()):
    raise SystemExit(1)

modes = json.loads(mode_path.read_text())
assert modes.get("final_production_sweep") is True, modes
assert modes.get("developed_state_diagnostics") is True, modes
assert modes.get("constitutive_change_from_v10_1_7") is False, modes
assert modes.get("geometry_source_feedback") is False, modes
assert modes.get("forward_spatial_source_field") is False, modes
assert modes.get("wake_shielding") is False, modes
assert math.isclose(float(modes["campaign_backstress_scale"]), back_scale, abs_tol=1e-14)
refresh = modes.get("campaign_refresh_scale", modes.get("campaign_refresh_length_scale"))
assert math.isclose(float(refresh), refresh_scale, abs_tol=1e-14)

summary = json.loads(summary_path.read_text())
assert summary and isinstance(summary, list), summary
row = summary[0]
minimum_advances = max(1, math.ceil(target_um / (da_m * 1.0e6) - 1.0e-12))
assert int(row["n_advances"]) >= minimum_advances, row
kc = float(row["Kc_first_MPa_sqrt_m"])
assert math.isfinite(kc) and 0.0 < kc <= kmax, row

audit = json.loads(audit_path.read_text())
records = audit.get("records", [])
assert records, audit
required = (
    "developed_state_active_count",
    "developed_state_mobile_count",
    "developed_state_retained_count",
    "developed_state_cumulative_emitted",
    "developed_state_cumulative_refreshed",
    "campaign_source_budget_total",
    "campaign_source_budget_remaining",
    "campaign_source_budget_consumed",
    "campaign_active_K_shield_effective_Pa_sqrt_m",
    "sigma_emission_backstress_Pa",
)
for record in records:
    for key in required:
        assert math.isfinite(float(record[key])), (key, record)

fired = [r for r in records if bool(r.get("fired", False))]
assert len(fired) >= minimum_advances, (len(fired), minimum_advances)
if run_mode == "plasticity_off":
    assert max(float(r["developed_state_active_count"]) for r in records) <= 1e-12
    assert max(float(r["developed_state_cumulative_emitted"]) for r in records) <= 1e-12
PY
}

for MODE in $MODES; do
  case "$MODE" in
    full)
      PLASTIC_FLAGS=(--tip-plasticity --active-shielding --signed-active-shielding)
      ;;
    plasticity_off)
      PLASTIC_FLAGS=(--no-tip-plasticity --no-active-shielding --signed-active-shielding)
      ;;
    active_shield_off)
      PLASTIC_FLAGS=(--tip-plasticity --no-active-shielding --signed-active-shielding)
      ;;
    *)
      echo "ERROR: MODE must be full, plasticity_off, or active_shield_off: $MODE" >&2
      exit 2
      ;;
  esac

  for CLASS in $CLASSES; do
    for T_K in $TEMPS; do
      OUTDIR="$OUTROOT/$MODE/$CLASS/T${T_K}_th${THETA}"
      mkdir -p "$OUTDIR"
      echo "========================================================================"
      echo "v10.1.7.1 final production: mode=$MODE class=$CLASS T=${T_K}K"
      echo "target=${TARGET_EXT_UM}um out=$OUTDIR"
      echo "========================================================================"

      status=COMPLETE
      if [[ "$FORCE" != 1 ]] && validate_case "$OUTDIR" "$MODE" >/dev/null 2>&1; then
        echo "SKIP validated complete case: $OUTDIR"
        status=EXISTING
      else
        rm -f "$OUTDIR/summary.json" "$OUTDIR/kinetic_tip_cell_audit_v101.json"
        if ! "$PYTHON_BIN" -m arrhenius_fracture.sharp_front_v10_1_7_1 \
          --mode 2d --material-class "$CLASS" --temperatures "$T_K" \
          --bulk-plasticity-mode tip_only --directional-j-mode root_signed \
          --tip-kinetics-mode moving_velocity --tip-source-model continuum \
          "${PLASTIC_FLAGS[@]}" \
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
          --no-wake-shielding \
          --crystal-aniso --crystal-compete --crystal-theta-deg "$THETA" \
          --crystal-material w --j-decomposition cluster \
          --max-fronts 1 --adaptive-events --adaptive-event-target "$EVENT_TARGET" \
          --print-every "$PRINT_EVERY" --save-snapshots "$SAVE_SNAPSHOTS" \
          --snapshot-by-crack-extension-um "$SNAPSHOT_BY_EXT_UM" \
          --out "$OUTDIR"; then
          status=FAILED
        elif ! validate_case "$OUTDIR" "$MODE"; then
          status=FAILED
        fi
      fi

      printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$MODE" "$CLASS" "$T_K" "$TARGET_EXT_UM" "$status" "$OUTDIR" >> "$MANIFEST"
      if [[ "$status" == FAILED ]]; then
        echo "ERROR: production case failed validation: $MODE/$CLASS/${T_K}K" >&2
        exit 1
      fi
    done
  done
done

"$PYTHON_BIN" scripts/analyze_v10_1_7_1_final_temperature_sweep.py \
  --root "$OUTROOT" \
  --classes $CLASSES \
  --temperatures $TEMPS \
  --modes $MODES \
  --theta "$THETA" \
  --checkpoint-advance-um "$("$PYTHON_BIN" - <<PY
print(float("$DA_CHECKPOINT_M") * 1.0e6)
PY
)"

echo "wrote $MANIFEST"
