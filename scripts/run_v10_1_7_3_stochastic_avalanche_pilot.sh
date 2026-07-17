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
TARGET_EXT_UM=${TARGET_EXT_UM:-200}
STEPS=${STEPS:-6000}
PRINT_EVERY=${PRINT_EVERY:-100}
OUTROOT=${OUTROOT:-runs/v10_1_7_3_stochastic_avalanche_DBTT_700K_200um_v1}
FORCE=${FORCE:-0}
HEARTBEAT_SECONDS=${HEARTBEAT_SECONDS:-60}

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
EVENT_MIN_FACTOR=${EVENT_MIN_FACTOR:-0.5}
EVENT_MAX_FACTOR=${EVENT_MAX_FACTOR:-4.0}
EVENT_SUBSEGMENT_FRACTION=${EVENT_SUBSEGMENT_FRACTION:-0.1}

export CAMPAIGN_BACKSTRESS_SCALE CAMPAIGN_REFRESH_SCALE
mkdir -p "$OUTROOT"
MANIFEST="$OUTROOT/stochastic_avalanche_pilot_manifest.tsv"
printf 'case_type\tmode\tseed\tclass\ttemperature_K\ttarget_ext_um\tstatus\toutdir\n' > "$MANIFEST"

stamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

report() {
  printf '[%s] %s\n' "$(stamp)" "$*"
}

CAMPAIGN_STATE=FAILED
CAMPAIGN_START=$(date +%s)
finish_report() {
  local rc=$?
  local elapsed=$(( $(date +%s) - CAMPAIGN_START ))
  if [[ "$CAMPAIGN_STATE" == COMPLETE && "$rc" -eq 0 ]]; then
    report "CAMPAIGN COMPLETE elapsed=${elapsed}s root=$OUTROOT"
  else
    report "CAMPAIGN FAILED rc=$rc elapsed=${elapsed}s root=$OUTROOT"
  fi
}
trap finish_report EXIT

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
step_paths = sorted(root.glob("steps_*K.csv"))
if not (
    summary_path.is_file()
    and audit_path.is_file()
    and mode_path.is_file()
    and len(step_paths) == 1
):
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
    assert modes.get("backend_semantic_identity") == "sharp_wake", modes
    assert modes.get("tip_following_remeshing_preserved") is True, modes

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
kinetic_path_m = max(
    float(r.get("kinetic_micro_advance_total_m", r.get("micro_advance_total_m", 0.0)))
    for r in records
)
assert math.isfinite(kinetic_path_m) and kinetic_path_m > 0.0, kinetic_path_m

# The 2-D stopping criterion is projected ligament-direction extension, not MPZ
# path arclength. Validate against the exact crack_extension_m column used by the
# solver instead of treating kinetic arclength as the stopping coordinate.
lines = [line.strip() for line in step_paths[0].read_text().splitlines() if line.strip()]
assert len(lines) >= 2, step_paths[0]
header = [token.strip() for token in lines[0].lstrip("# ").split(",")]
idx_ext = header.index("crack_extension_m")
projected_values = []
for line in lines[1:]:
    values = line.split(",")
    if len(values) <= idx_ext:
        continue
    value = float(values[idx_ext])
    if math.isfinite(value):
        projected_values.append(value)
assert projected_values, step_paths[0]
projected_extension_m = max(projected_values)
assert projected_extension_m + 1.0e-12 >= target_um * 1.0e-6, (
    projected_extension_m * 1.0e6,
    target_um,
)
# A path arclength cannot be shorter than its positive x projection.
projection_tol = max(5.0e-9, 5.0e-6 * max(kinetic_path_m, projected_extension_m))
assert kinetic_path_m + projection_tol >= projected_extension_m, (
    kinetic_path_m,
    projected_extension_m,
)

equivalent_checkpoints = float(row.get("n_advances", 0))
if case_type == "fixed_original":
    thresholds = [float(r["hazard_last_completed_threshold"]) for r in fired]
    assert all(abs(x - 1.0) <= 1.0e-14 for x in thresholds), thresholds
else:
    lengths = [float(r["avalanche_event_advance_m"]) for r in fired]
    assert all(math.isfinite(x) and x > 0.0 for x in lengths), lengths

    geometry_path = root / "stochastic_avalanche_geometry_events.json"
    assert geometry_path.is_file(), geometry_path
    geometry = json.loads(geometry_path.read_text())
    # This is the actual stochastic event count: every completed cleavage renewal
    # must produce exactly one geometry commit and one fired audit record.
    assert len(geometry) == len(fired), (len(geometry), len(fired))

    geometry_lengths = []
    for index, (event, kinetic_length) in enumerate(zip(geometry, lengths)):
        requested = float(event["requested_event_advance_m"])
        realized = float(event["event_advance_m"])
        geometry_lengths.append(realized)
        assert int(event.get("realized_geometry_commits", 0)) == 1, event
        assert event.get("geometry_realization") == "single_checked_outer_commit", event
        assert event.get("backend_semantic_identity") == "sharp_wake", event
        assert event.get("tip_following_remeshing_preserved") is True, event
        assert event.get("driver_endpoint_synchronized") is True, event
        tolerance = max(1.0e-9, 0.05 * requested)
        assert abs(realized - requested) <= tolerance, event
        assert abs(realized - kinetic_length) <= max(1.0e-9, 1.0e-6 * realized), (
            event,
            kinetic_length,
        )
        if index > 0:
            previous = geometry[index - 1]
            continuity = math.hypot(
                float(event["x0"]) - float(previous["x1"]),
                float(event["y0"]) - float(previous["y1"]),
            )
            assert continuity <= 5.0e-9, (index, continuity, previous, event)

    geometry_path_m = sum(geometry_lengths)
    path_tol = max(5.0e-9, 5.0e-6 * max(geometry_path_m, kinetic_path_m))
    assert abs(geometry_path_m - kinetic_path_m) <= path_tol, (
        geometry_path_m,
        kinetic_path_m,
    )

    geometry_projected_m = float(geometry[-1]["x1"]) - float(geometry[0]["x0"])
    assert abs(geometry_projected_m - projected_extension_m) <= projection_tol, (
        geometry_projected_m,
        projected_extension_m,
    )
    assert geometry_path_m + path_tol >= geometry_projected_m, (
        geometry_path_m,
        geometry_projected_m,
    )

    # Legacy summary semantics: n_advances is the rounded path length measured in
    # nominal da_phys units, not the number of variable-length events. Preserve
    # that field and add an explicit event count so the two cannot be conflated.
    equivalent_exact = geometry_path_m / da_m
    equivalent_rounded = int(round(equivalent_exact))
    assert int(row.get("n_advances", -1)) == equivalent_rounded, (
        row.get("n_advances"),
        equivalent_exact,
        len(geometry),
    )
    equivalent_checkpoints = equivalent_exact

    expected_summary = {
        "n_geometry_events": int(len(geometry)),
        "n_equivalent_checkpoints_exact": float(equivalent_exact),
        "n_equivalent_checkpoints_rounded": int(equivalent_rounded),
        "nominal_checkpoint_length_m": float(da_m),
        "geometry_path_length_m": float(geometry_path_m),
        "geometry_projected_extension_m": float(geometry_projected_m),
        "n_advances_semantics": "rounded_path_length_over_nominal_checkpoint",
        "n_geometry_events_semantics": "accepted_cleavage_renewals_and_geometry_commits",
    }
    changed = any(row.get(key) != value for key, value in expected_summary.items())
    row.update(expected_summary)
    if changed:
        # Repair completed outputs generated immediately before these explicit
        # metadata fields were introduced. No FEM rerun is needed when all path,
        # endpoint, event, and kinetic-state checks above already pass.
        summary_path.write_text(json.dumps(summary, indent=2))

    assert int(row["n_geometry_events"]) == len(geometry), row
    assert math.isclose(
        float(row["n_equivalent_checkpoints_exact"]),
        equivalent_exact,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ), row

    if case_type == "segmented_deterministic":
        assert all(abs(x - da_m) <= 1.0e-12 * max(da_m, 1.0) for x in lengths), lengths
    else:
        norm = min_factor + math.exp(-min_factor) - math.exp(-max_factor)
        lo = da_m * min_factor / norm * 0.99
        hi = da_m * max_factor / norm * 1.01
        assert all(lo <= x <= hi for x in lengths), (lo, hi, lengths)
        assert any(abs(x - da_m) > 1.0e-3 * da_m for x in lengths), lengths
        assert all(r.get("stochastic_avalanche_length_enabled") is True for r in records)

print(
    "VALIDATION "
    f"case={case_type} seed={seed} events={len(fired)} "
    f"equivalent_checkpoints={equivalent_checkpoints:.3f} "
    f"projected={projected_extension_m*1.0e6:.3f}um "
    f"path={kinetic_path_m*1.0e6:.3f}um"
)
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

run_solver_with_heartbeat() {
  local case_type=$1
  local mode=$2
  local seed=$3
  local entry=$4
  local length_mode=$5
  local outdir=$6
  shift 6
  local -a flags=("$@")

  env \
    PYTHONUNBUFFERED=1 \
    CLEAVAGE_HAZARD_MODE="$mode" \
    CLEAVAGE_HAZARD_SEED="$seed" \
    CLEAVAGE_EVENT_LENGTH_MODE="$length_mode" \
    CLEAVAGE_EVENT_MIN_FACTOR="$EVENT_MIN_FACTOR" \
    CLEAVAGE_EVENT_MAX_FACTOR="$EVENT_MAX_FACTOR" \
    CLEAVAGE_EVENT_SUBSEGMENT_FRACTION="$EVENT_SUBSEGMENT_FRACTION" \
    "$PYTHON_BIN" -u -m "$entry" "${flags[@]}" --out "$outdir" &
  local solver_pid=$!
  local start=$(date +%s)
  report "SOLVER PID=$solver_pid case=$case_type seed=$seed"

  while kill -0 "$solver_pid" 2>/dev/null; do
    sleep "$HEARTBEAT_SECONDS"
    if kill -0 "$solver_pid" 2>/dev/null; then
      local elapsed=$(( $(date +%s) - start ))
      report "HEARTBEAT case=$case_type seed=$seed pid=$solver_pid elapsed=${elapsed}s"
    fi
  done

  set +e
  wait "$solver_pid"
  local rc=$?
  set -e
  return "$rc"
}

run_case() {
  local case_type=$1
  local mode=$2
  local seed=$3
  local entry=$4
  local length_mode=$5
  local outdir=$6
  local status=COMPLETE
  local case_start=$(date +%s)

  report "CASE START case=$case_type mode=$mode seed=$seed class=$CLASS T=${TEMP_K}K target=${TARGET_EXT_UM}um"
  report "CASE OUTDIR $outdir"

  if [[ "$FORCE" != 1 ]] && validate_case "$outdir" "$case_type" "$mode" "$seed" >/dev/null 2>&1; then
    status=EXISTING
    report "CASE SKIP validated-complete case=$case_type seed=$seed"
  else
    # Remove every stale product from a failed or invalid case. Never retain
    # geometry-event diagnostics from either the superseded 20-um multiplication
    # bug or the variable-event/front-endpoint desynchronization bug.
    rm -rf "$outdir"
    mkdir -p "$outdir"
    local -a FLAGS=()
    while IFS= read -r flag; do
      FLAGS+=("$flag")
    done < <(common_flags)

    if ! run_solver_with_heartbeat \
      "$case_type" "$mode" "$seed" "$entry" "$length_mode" "$outdir" \
      "${FLAGS[@]}"; then
      status=FAILED
    elif ! validate_case "$outdir" "$case_type" "$mode" "$seed"; then
      status=FAILED
    fi
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$case_type" "$mode" "$seed" "$CLASS" "$TEMP_K" "$TARGET_EXT_UM" "$status" "$outdir" \
    >> "$MANIFEST"

  local elapsed=$(( $(date +%s) - case_start ))
  if [[ "$status" == FAILED ]]; then
    report "CASE FAILED case=$case_type seed=$seed elapsed=${elapsed}s"
    exit 1
  fi
  report "CASE COMPLETE case=$case_type seed=$seed status=$status elapsed=${elapsed}s"
}

N_SEEDS=0
for _seed in $SEEDS; do N_SEEDS=$((N_SEEDS + 1)); done
report "CAMPAIGN START class=$CLASS T=${TEMP_K}K target=${TARGET_EXT_UM}um stochastic_seeds=$N_SEEDS"
report "CONFIG mean_event=${DA_CHECKPOINT_M}m factor_bounds=[$EVENT_MIN_FACTOR,$EVENT_MAX_FACTOR] heartbeat=${HEARTBEAT_SECONDS}s"

run_case fixed_original deterministic 0 \
  arrhenius_fracture.sharp_front_v10_1_7_2 fixed \
  "$OUTROOT/fixed_original/T${TEMP_K}_th${THETA}"

# This is retained as the deterministic control for the variable-event backend.
# The backend now makes one checked 5-um geometry commit; it does not claim ten
# re-equilibrated subincrements.
run_case segmented_deterministic deterministic 0 \
  arrhenius_fracture.sharp_front_v10_1_7_3 fixed \
  "$OUTROOT/segmented_deterministic/T${TEMP_K}_th${THETA}"

for SEED in $SEEDS; do
  run_case stochastic_avalanche exponential "$SEED" \
    arrhenius_fracture.sharp_front_v10_1_7_3 threshold_scaled \
    "$OUTROOT/stochastic_avalanche/seed_${SEED}/T${TEMP_K}_th${THETA}"
done

report "ANALYSIS START"
"$PYTHON_BIN" -u scripts/analyze_v10_1_7_3_stochastic_avalanche_pilot.py \
  --root "$OUTROOT" --class "$CLASS" --temperature "$TEMP_K" \
  --seeds $SEEDS --theta "$THETA" --base-checkpoint-um "$("$PYTHON_BIN" - <<PY
print(float("$DA_CHECKPOINT_M") * 1.0e6)
PY
)"
report "ANALYSIS COMPLETE"
report "MANIFEST $MANIFEST"
CAMPAIGN_STATE=COMPLETE
