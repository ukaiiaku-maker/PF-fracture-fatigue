#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN=${PYTHON_BIN:-python}
TEMPS=${TEMPS:-"300 1100"}
INTERACTION_SCALES=${INTERACTION_SCALES:-"1 2"}
RETENTION_SCALES=${RETENTION_SCALES:-"1 3"}
TARGET_EXT_UM=${TARGET_EXT_UM:-100}
STEPS=${STEPS:-4000}
PRINT_EVERY=${PRINT_EVERY:-100}
OUTROOT=${OUTROOT:-runs/v10_1_8_forward_interaction_zone_100um_4set_v1}
FORCE=${FORCE:-0}

NX=${NX:-48}
NY=${NY:-96}
TIP_H_FINE=${TIP_H_FINE:-5e-7}
TIP_RATIO=${TIP_RATIO:-1.2}
DU=${DU:-2e-7}
DT=${DT:-8.4}
N_STAGGER=${N_STAGGER:-2}
DA_CHECKPOINT_M=${DA_CHECKPOINT_M:-5e-6}
MPZ_LENGTH_UM=${MPZ_LENGTH_UM:-150}
MPZ_N_BINS=${MPZ_N_BINS:-300}
WAKE_LENGTH_UM=${WAKE_LENGTH_UM:-100}
WAKE_N_BINS=${WAKE_N_BINS:-0}
THETA=${THETA:-45}
EVENT_TARGET=${EVENT_TARGET:-0.05}
SAVE_SNAPSHOTS=${SAVE_SNAPSHOTS:-0}
SNAPSHOT_BY_EXT_UM=${SNAPSHOT_BY_EXT_UM:-10}
PACKET_LENGTH_M=${PACKET_LENGTH_M:-2.5e-10}
MOBILE_SHIELD_FRACTION=${MOBILE_SHIELD_FRACTION:-1.0}
KINETIC_MAX_ACTION_SUBSTEP=${KINETIC_MAX_ACTION_SUBSTEP:-0.01}
KINETIC_MAX_TRANSLATION_SUBSTEP_M=${KINETIC_MAX_TRANSLATION_SUBSTEP_M:-5e-8}
K_FIRST_MAX_MPA_SQRT_M=${K_FIRST_MAX_MPA_SQRT_M:-200}

mkdir -p "$OUTROOT"
MANIFEST="$OUTROOT/forward_zone_manifest.tsv"
printf 'kind\tinteraction_scale\tretention_scale\ttemperature_K\tstatus\toutdir\n' > "$MANIFEST"

scale_tag() {
  printf '%s' "$1" | sed 's/-/m/g; s/\./p/g'
}

run_case() {
  local kind=$1
  local length_scale=$2
  local retention_scale=$3
  local temp=$4
  local outdir=$5
  shift 5
  local flags=("$@")

  mkdir -p "$outdir"
  echo "========================================================================"
  echo "v10.1.8 forward-zone: kind=$kind T=${temp}K Lscale=$length_scale retention=$retention_scale"
  echo "target=${TARGET_EXT_UM}um out=$outdir"
  echo "========================================================================"

  local status=COMPLETE
  if [[ "$FORCE" != 1 && -s "$outdir/summary.json" && -s "$outdir/kinetic_tip_cell_audit_v101.json" ]]; then
    echo "SKIP existing complete-looking case: $outdir"
    status=EXISTING
  else
    if ! CAMPAIGN_BACKSTRESS_SCALE=1 \
         CAMPAIGN_REFRESH_SCALE=1 \
         FORWARD_INTERACTION_LENGTH_SCALE="$length_scale" \
         FORWARD_RETENTION_SCALE="$retention_scale" \
         "$PYTHON_BIN" -m arrhenius_fracture.sharp_front_v10_1_8 \
      --mode 2d --material-class DBTT --temperatures "$temp" \
      --bulk-plasticity-mode tip_only --directional-j-mode root_signed \
      --tip-kinetics-mode moving_velocity --tip-source-model continuum \
      "${flags[@]}" \
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
      --out "$outdir"; then
      status=FAILED
    fi
  fi

  if [[ "$status" != FAILED ]]; then
    if ! "$PYTHON_BIN" - "$outdir" "$kind" "$length_scale" "$retention_scale" \
      "$TARGET_EXT_UM" "$DA_CHECKPOINT_M" "$K_FIRST_MAX_MPA_SQRT_M" <<'PY'
import json, math, pathlib, sys
root = pathlib.Path(sys.argv[1])
kind = sys.argv[2]
length_scale = float(sys.argv[3])
retention_scale = float(sys.argv[4])
target_um = float(sys.argv[5])
da_m = float(sys.argv[6])
kmax = float(sys.argv[7])

modes = json.loads((root / "v10_1_driver_modes.json").read_text())
assert modes["tip_source_model"] == "forward_interaction_zone", modes
assert modes["wake_shielding"] is False, modes
assert modes["scalar_uniform_source_refresh"] is False, modes
assert math.isclose(float(modes["forward_interaction_length_scale"]), length_scale, abs_tol=1e-14)
assert math.isclose(float(modes["forward_retention_scale"]), retention_scale, abs_tol=1e-14)

summary = json.loads((root / "summary.json").read_text())[0]
minimum_advances = max(1, math.ceil(target_um / (da_m * 1e6) - 1e-12))
assert int(summary["n_advances"]) >= minimum_advances, summary
kc = float(summary["Kc_first_MPa_sqrt_m"])
assert math.isfinite(kc) and 0.0 < kc <= kmax, summary

audit = json.loads((root / "kinetic_tip_cell_audit_v101.json").read_text())
records = audit.get("records", [])
assert records, audit
required = (
    "forward_interaction_length_m",
    "forward_source_capacity_total",
    "forward_source_available_total",
    "forward_source_cumulative_consumed",
    "forward_source_cumulative_inflow",
    "forward_active_mobile_count",
    "forward_active_retained_count",
    "sigma_emission_backstress_Pa",
    "campaign_active_K_shield_effective_Pa_sqrt_m",
)
for key in required:
    assert all(math.isfinite(float(r[key])) for r in records), key
assert max(abs(float(r.get("kinetic_wake_K_shield_signed_Pa_sqrt_m", 0.0))) for r in records) <= 1e-12
if kind == "baseline":
    assert max(float(r["forward_source_cumulative_consumed"]) for r in records) <= 1e-12
    assert max(float(r["forward_active_mobile_count"]) for r in records) <= 1e-12
    assert max(float(r["forward_active_retained_count"]) for r in records) <= 1e-12
PY
    then
      status=FAILED
    fi
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$kind" "$length_scale" "$retention_scale" "$temp" "$status" "$outdir" >> "$MANIFEST"
  if [[ "$status" == FAILED ]]; then
    echo "ERROR: forward-zone case failed: $kind L=$length_scale retention=$retention_scale T=$temp" >&2
    exit 1
  fi
}

for T_K in $TEMPS; do
  run_case baseline 1 1 "$T_K" \
    "$OUTROOT/baseline/T${T_K}_th${THETA}" \
    --no-tip-plasticity --no-active-shielding --signed-active-shielding
done

for LENGTH in $INTERACTION_SCALES; do
  for RETENTION in $RETENTION_SCALES; do
    LTAG=$(scale_tag "$LENGTH")
    RTAG=$(scale_tag "$RETENTION")
    for T_K in $TEMPS; do
      run_case full "$LENGTH" "$RETENTION" "$T_K" \
        "$OUTROOT/full/L${LTAG}_R${RTAG}/T${T_K}_th${THETA}" \
        --tip-plasticity --active-shielding --signed-active-shielding
    done
  done
done

"$PYTHON_BIN" scripts/analyze_v10_1_8_forward_zone.py \
  --root "$OUTROOT" \
  --temperatures $TEMPS \
  --interaction-scales $INTERACTION_SCALES \
  --retention-scales $RETENTION_SCALES \
  --theta "$THETA"

echo "wrote $MANIFEST"
