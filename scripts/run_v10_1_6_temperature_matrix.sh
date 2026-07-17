#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN=${PYTHON_BIN:-python}
CLASSES=${CLASSES:-"ceramic weakT DBTT"}
TEMPS=${TEMPS:-"300 700 1100"}
MODES=${MODES:-"full plasticity_off"}
TARGET_EXT_UM=${TARGET_EXT_UM:-50}
STEPS=${STEPS:-2000}
PRINT_EVERY=${PRINT_EVERY:-50}
OUTROOT=${OUTROOT:-runs/v10_1_6_emergent_temperature_matrix_50um_v1}
FORCE=${FORCE:-0}

# These scales are common to every class and temperature.  The runner does not
# permit a per-temperature source budget, refresh law, back stress, shielding
# amplitude, or blunting coefficient.
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
SNAPSHOT_BY_EXT_UM=${SNAPSHOT_BY_EXT_UM:-5}
PACKET_LENGTH_M=${PACKET_LENGTH_M:-2.5e-10}
MOBILE_SHIELD_FRACTION=${MOBILE_SHIELD_FRACTION:-1.0}
KINETIC_MAX_ACTION_SUBSTEP=${KINETIC_MAX_ACTION_SUBSTEP:-0.01}
KINETIC_MAX_TRANSLATION_SUBSTEP_M=${KINETIC_MAX_TRANSLATION_SUBSTEP_M:-5e-8}

DBTT_LOW_MAX=${DBTT_LOW_MAX:-0.5}
DBTT_MIN_EMERGENCE=${DBTT_MIN_EMERGENCE:-1.0}
FLAT_MAX_SPAN=${FLAT_MAX_SPAN:-1.0}

export CAMPAIGN_BACKSTRESS_SCALE CAMPAIGN_REFRESH_SCALE
mkdir -p "$OUTROOT"
MANIFEST="$OUTROOT/temperature_matrix_manifest.tsv"
printf 'mode\tclass\ttemperature_K\tbackstress_scale\trefresh_scale\ttarget_ext_um\tstatus\toutdir\n' > "$MANIFEST"

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
      echo "ERROR: mode must be full, plasticity_off, or active_shield_off: $MODE" >&2
      exit 2
      ;;
  esac

  for CLASS in $CLASSES; do
    for T_K in $TEMPS; do
      OUTDIR="$OUTROOT/$MODE/$CLASS/T${T_K}_th${THETA}"
      mkdir -p "$OUTDIR"
      echo "========================================================================"
      echo "v10.1.6 temperature matrix: mode=$MODE class=$CLASS T=${T_K}K"
      echo "common backstress_scale=$CAMPAIGN_BACKSTRESS_SCALE refresh_scale=$CAMPAIGN_REFRESH_SCALE"
      echo "target=${TARGET_EXT_UM}um out=$OUTDIR"
      echo "========================================================================"

      if [[ "$FORCE" != 1 && -s "$OUTDIR/summary.json" && -s "$OUTDIR/kinetic_tip_cell_audit_v101.json" ]]; then
        echo "SKIP existing complete-looking case: $OUTDIR"
        status=EXISTING
      else
        status=COMPLETE
        if ! "$PYTHON_BIN" -m arrhenius_fracture.sharp_front_v10_1_6 \
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
        fi
      fi

      if [[ "$status" != FAILED ]]; then
        if ! "$PYTHON_BIN" - \
          "$OUTDIR/v10_1_driver_modes.json" \
          "$OUTDIR/summary.json" \
          "$OUTDIR/kinetic_tip_cell_audit_v101.json" \
          "$TARGET_EXT_UM" \
          "$DA_CHECKPOINT_M" \
          "$K_FIRST_MAX_MPA_SQRT_M" \
          "$MODE" \
          "$CAMPAIGN_BACKSTRESS_SCALE" \
          "$CAMPAIGN_REFRESH_SCALE" <<'PY'
import json, math, pathlib, sys

mode_path = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
audit_path = pathlib.Path(sys.argv[3])
target_um = float(sys.argv[4])
da_m = float(sys.argv[5])
kmax = float(sys.argv[6])
run_mode = sys.argv[7]
back_scale = float(sys.argv[8])
refresh_scale = float(sys.argv[9])

modes = json.loads(mode_path.read_text())
assert modes["tip_source_model"] == "campaign_calibrated", modes
assert modes["source_recovery_time"] == "none_while_stationary", modes
assert modes["manifest_K_shield_cap_enabled"] is True, modes
assert math.isclose(float(modes["campaign_backstress_scale"]), back_scale, rel_tol=0.0, abs_tol=1e-14), modes
assert math.isclose(float(modes["campaign_refresh_scale"]), refresh_scale, rel_tol=0.0, abs_tol=1e-14), modes

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
assert audit["campaign_calibration"]["temporal_source_recycling"] is False, audit
assert math.isclose(float(audit["campaign_calibration"]["backstress_scale"]), back_scale, rel_tol=0.0, abs_tol=1e-14)
assert math.isclose(float(audit["campaign_calibration"]["refresh_length_scale"]), refresh_scale, rel_tol=0.0, abs_tol=1e-14)

required = (
    "campaign_source_budget_total",
    "campaign_source_budget_remaining",
    "campaign_source_budget_consumed",
    "campaign_active_K_shield_effective_Pa_sqrt_m",
    "campaign_active_K_shield_cap_Pa_sqrt_m",
    "sigma_opening_tip_Pa",
    "sigma_cleave_eff_Pa",
    "sigma_emission_backstress_Pa",
    "sigma_emission_effective_Pa",
)
for record in records:
    for key in required:
        value = float(record[key])
        assert math.isfinite(value), (key, value, record)
    budget = float(record["campaign_source_budget_total"])
    remaining = float(record["campaign_source_budget_remaining"])
    consumed = float(record["campaign_source_budget_consumed"])
    assert -1e-10 <= remaining <= budget + 1e-10
    assert -1e-10 <= consumed <= budget + 1e-10
    cap = float(record["campaign_active_K_shield_cap_Pa_sqrt_m"])
    eff = abs(float(record["campaign_active_K_shield_effective_Pa_sqrt_m"]))
    assert eff <= cap + max(1e-6, 1e-10 * max(cap, 1.0))
    assert record["campaign_temporal_source_recycling"] is False

if run_mode == "plasticity_off":
    assert max(float(r.get("active_mobile", 0.0)) + float(r.get("active_retained", 0.0)) for r in records) <= 1e-12
    assert max(float(r["campaign_source_budget_consumed"]) for r in records) <= 1e-10
PY
        then
          status=FAILED
        fi
      fi

      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$MODE" "$CLASS" "$T_K" "$CAMPAIGN_BACKSTRESS_SCALE" \
        "$CAMPAIGN_REFRESH_SCALE" "$TARGET_EXT_UM" "$status" "$OUTDIR" >> "$MANIFEST"

      if [[ "$status" == FAILED ]]; then
        echo "ERROR: matrix case failed: $MODE/$CLASS/${T_K}K; see $OUTDIR" >&2
        exit 1
      fi
    done
  done
done

"$PYTHON_BIN" scripts/analyze_v10_1_6_temperature_matrix.py \
  --root "$OUTROOT" \
  --classes $CLASSES \
  --temperatures $TEMPS \
  --modes $MODES \
  --theta "$THETA" \
  --dbtt-low-max "$DBTT_LOW_MAX" \
  --dbtt-min-emergence "$DBTT_MIN_EMERGENCE" \
  --flat-max-span "$FLAT_MAX_SPAN"

echo "wrote $MANIFEST"
