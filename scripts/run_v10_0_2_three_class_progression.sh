#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN=${PYTHON_BIN:-python}
TEMPS=${TEMPS:-700}
CLASSES=${CLASSES:-"ceramic weakT DBTT"}
TARGET_EXT_UM=${TARGET_EXT_UM:-50}
STEPS=${STEPS:-50000}
OUTROOT_BASE=${OUTROOT_BASE:-runs/v10_0_2_three_class_progression}

NX=${NX:-48}
NY=${NY:-96}
TIP_H_FINE=${TIP_H_FINE:-5e-7}
TIP_RATIO=${TIP_RATIO:-1.2}
DU=${DU:-2e-7}
DT=${DT:-8.4}
N_STAGGER=${N_STAGGER:-2}
DA_PHYS_M=${DA_PHYS_M:-5e-6}

MPZ_LENGTH_UM=${MPZ_LENGTH_UM:-100}
MPZ_N_BINS=${MPZ_N_BINS:-200}
WAKE_LENGTH_UM=${WAKE_LENGTH_UM:-100}
WAKE_N_BINS=${WAKE_N_BINS:-0}
WAKE_SHIELDING=${WAKE_SHIELDING:-1}
WAKE_SHIELD_PROJECTION=${WAKE_SHIELD_PROJECTION:-1}

THETA=${THETA:-45}
MAX_FRONTS=${MAX_FRONTS:-1}
EVENT_TARGET=${EVENT_TARGET:-0.15}
PRINT_EVERY=${PRINT_EVERY:-25}
SAVE_SNAPSHOTS=${SAVE_SNAPSHOTS:-10}
SNAPSHOT_BY_EXT_UM=${SNAPSHOT_BY_EXT_UM:-5}

"$PYTHON_BIN" - "$DA_PHYS_M" <<'PY'
import math, sys
value = float(sys.argv[1])
if not math.isclose(value, 5.0e-6, rel_tol=0.0, abs_tol=1.0e-15):
    raise SystemExit(
        "v10.0.2.1 requires DA_PHYS_M=5e-6. "
        "Physical renewal length and numerical geometry substep are not yet separated."
    )
PY

if [[ "$MAX_FRONTS" != "1" ]]; then
  echo "ERROR: the material-progression gate is single-front. Use the branching gate later." >&2
  exit 2
fi

case "$WAKE_SHIELDING" in
  0|1) ;;
  *) echo "ERROR: WAKE_SHIELDING must be 0 or 1." >&2; exit 2 ;;
esac

mkdir -p "$OUTROOT_BASE"
MANIFEST="$OUTROOT_BASE/progression_manifest.tsv"
if [[ ! -f "$MANIFEST" ]]; then
  printf 'temperature_K\tclass\twake_shielding\ttarget_ext_um\tda_phys_m\tstatus\toutdir\n' > "$MANIFEST"
fi

for T_K in $TEMPS; do
  for CLASS in $CLASSES; do
    OUTDIR="$OUTROOT_BASE/T${T_K}/wake_${WAKE_SHIELDING}/${CLASS}/T${T_K}_th${THETA}"
    mkdir -p "$OUTDIR"

    wake_args=(
      --wake-length-um "$WAKE_LENGTH_UM"
      --wake-n-bins "$WAKE_N_BINS"
      --wake-shield-projection "$WAKE_SHIELD_PROJECTION"
    )
    if [[ "$WAKE_SHIELDING" == "1" ]]; then
      wake_args+=(--wake-shielding)
    else
      wake_args+=(--no-wake-shielding)
    fi

    echo "========================================================================"
    echo "v10.0.2.1 progression: class=$CLASS T=${T_K}K target=${TARGET_EXT_UM}um wake=$WAKE_SHIELDING"
    echo "out=$OUTDIR"
    echo "========================================================================"

    status=COMPLETE
    if ! "$PYTHON_BIN" -m arrhenius_fracture.sharp_front_v10_1 \
      --mode 2d --material-class "$CLASS" --temperatures "$T_K" \
      --bulk-plasticity-mode tip_only --directional-j-mode root_signed \
      --steps "$STEPS" --nx "$NX" --ny "$NY" \
      --dU "$DU" --dt "$DT" --n-stagger "$N_STAGGER" \
      --tip-h-fine "$TIP_H_FINE" --tip-ratio "$TIP_RATIO" \
      --da-phys "$DA_PHYS_M" --target-crack-extension-um "$TARGET_EXT_UM" \
      --mpz-length-um "$MPZ_LENGTH_UM" --mpz-n-bins "$MPZ_N_BINS" \
      "${wake_args[@]}" \
      --crystal-aniso --crystal-compete --crystal-theta-deg "$THETA" \
      --crystal-material w --j-decomposition cluster \
      --max-fronts 1 --adaptive-events --adaptive-event-target "$EVENT_TARGET" \
      --print-every "$PRINT_EVERY" --save-snapshots "$SAVE_SNAPSHOTS" \
      --snapshot-by-crack-extension-um "$SNAPSHOT_BY_EXT_UM" \
      --out "$OUTDIR"; then
      status=FAILED
    fi

    "$PYTHON_BIN" - "$OUTDIR/run_args.json" "$WAKE_SHIELDING" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
expected = bool(int(sys.argv[2]))
data = json.loads(path.read_text())
actual = bool(data.get("wake_shielding"))
if actual != expected:
    raise SystemExit(f"wake routing audit failed: expected {expected}, run_args has {actual}: {path}")
PY

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$T_K" "$CLASS" "$WAKE_SHIELDING" "$TARGET_EXT_UM" "$DA_PHYS_M" "$status" "$OUTDIR" \
      >> "$MANIFEST"

    if [[ "$status" != COMPLETE ]]; then
      echo "ERROR: $CLASS at ${T_K}K failed; see $OUTDIR" >&2
      exit 1
    fi
  done
done

echo "wrote $MANIFEST"
