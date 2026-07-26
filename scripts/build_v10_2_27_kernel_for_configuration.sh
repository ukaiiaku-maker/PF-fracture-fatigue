#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
CONDA_ENV=${CONDA_ENV:-arrhenius-sharp-front-v10}
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
  echo "ERROR: activate conda environment '$CONDA_ENV'" >&2
  exit 2
fi

: "${V10227_KERNEL_CONFIGURATION:?Resolver must set V10227_KERNEL_CONFIGURATION}"
: "${V10227_KERNEL_CACHE_DIR:?Resolver must set V10227_KERNEL_CACHE_DIR}"
: "${V10227_KERNEL_FAMILY_OUT:?Resolver must set V10227_KERNEL_FAMILY_OUT}"
: "${V10227_KERNEL_TARGET_EXTENSION_UM:?Resolver must set V10227_KERNEL_TARGET_EXTENSION_UM}"

CONFIG="$V10227_KERNEL_CONFIGURATION"
CACHE_DIR="$V10227_KERNEL_CACHE_DIR"
FAMILY_OUT="$V10227_KERNEL_FAMILY_OUT"
TARGET_EXT_UM="$V10227_KERNEL_TARGET_EXTENSION_UM"
REQUIRED_EXT_UM=${V10227_KERNEL_REQUIRED_MAX_EXTENSION_UM:-$TARGET_EXT_UM}
DA_PHYS_UM=${V10227_KERNEL_DA_PHYS_UM:-${DA_PHYS_UM:-5}}
EVENT_MINIMUM_FACTOR=${V10227_KERNEL_EVENT_MINIMUM_FACTOR:-${CLEAVAGE_EVENT_MIN_FACTOR:-0.5}}
EVENT_MAXIMUM_FACTOR=${V10227_KERNEL_EVENT_MAXIMUM_FACTOR:-${CLEAVAGE_EVENT_MAX_FACTOR:-4.0}}
MARGIN_EVENTS=${V10227_KERNEL_MARGIN_EVENTS:-${KERNEL_MARGIN_EVENTS:-1}}
MAX_ITERATIONS=${KERNEL_SELF_CONSISTENCY_MAX_ITERATIONS:-4}
MAX_RELATIVE_CHANGE=${KERNEL_SELF_CONSISTENCY_MAX_RELATIVE_CHANGE:-0.02}
MAX_ABSOLUTE_CHANGE=${KERNEL_SELF_CONSISTENCY_MAX_ABSOLUTE_CHANGE_PA_SQRT_M_PER_LINE:-100}
MAX_EXTENSION_CHANGE_UM=${KERNEL_SELF_CONSISTENCY_MAX_EXTENSION_CHANGE_UM:-1}
MAX_NORMALIZATION_CHANGE=${KERNEL_SELF_CONSISTENCY_MAX_NORMALIZATION_RELATIVE_CHANGE:-1e-6}

read -r THETA BRANCHING_MODE MAX_FRONTS CAPTURE_TEMPERATURE_K C11 C12 C44 PZ_LENGTH_M < <(
  "$PYTHON_BIN" - "$CONFIG" <<'PY'
import json
import sys
payload = json.loads(open(sys.argv[1]).read())
temperature = payload.get("temperature_K") if payload.get("temperature_dependent_mechanics") else 700.0
print(
    payload["theta_deg"],
    payload["branching_mode"],
    payload["maximum_fronts"],
    temperature,
    payload["crystal_C11_Pa"],
    payload["crystal_C12_Pa"],
    payload["crystal_C44_Pa"],
    payload["process_zone_length_m"],
)
PY
)

if [[ "$BRANCHING_MODE" != single_front || "$MAX_FRONTS" != 1 ]]; then
  echo "ERROR: fixed-extension atlas builder cannot serve branching topology." >&2
  echo "Register a topology_cached or direct_fem builder for this configuration." >&2
  exit 4
fi
if ! [[ "$MAX_ITERATIONS" =~ ^[0-9]+$ ]] || (( MAX_ITERATIONS < 2 )); then
  echo "ERROR: KERNEL_SELF_CONSISTENCY_MAX_ITERATIONS must be an integer >= 2" >&2
  exit 2
fi

export V10227_CRYSTAL_C11_PA="$C11"
export V10227_CRYSTAL_C12_PA="$C12"
export V10227_CRYSTAL_C44_PA="$C44"

sha256_file() {
  "$PYTHON_BIN" - "$1" <<'PY'
import hashlib
import pathlib
import sys
print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
PY
}

capture_audits() {
  local snapshot_root=$1
  echo "AUDIT exact active endpoint resolution" >&2
  "$PYTHON_BIN" scripts/check_v10_2_27_active_endpoint_resolution.py \
    --snapshot-root "$snapshot_root" \
    --mechanical-config "$CONFIG"
  echo "AUDIT accepted production capture physics" >&2
  "$PYTHON_BIN" scripts/check_v10_2_27_capture_physics_contract.py \
    --snapshot-root "$snapshot_root" \
    --mechanical-config "$CONFIG"
}

build_load_invariance() {
  local snapshot_root=$1
  local load_root=$2
  rm -rf "$load_root"
  mkdir -p "$load_root"
  local found=0
  while IFS= read -r snapshot_json; do
    local state_root state destination
    state_root=$(dirname "$snapshot_json")
    state=$(basename "$state_root")
    destination="$load_root/$state"
    found=$((found + 1))
    echo "RECALCULATE load invariance endpoints: $state" >&2
    "$PYTHON_BIN" scripts/evaluate_v10_2_14_active_load_invariance.py \
      --snapshot "$state_root" \
      --outroot "$destination" \
      --load-scales 0.5 1.0 1.5 \
      --magnitudes 0.25 0.50 \
      --linearity-tolerance 0.03 \
      --load-invariance-tolerance 0.05 \
      --minimum-residual-stiffness-fraction 0.001 \
      --minimum-station-spacing-m "$PZ_LENGTH_M"
  done < <(find "$snapshot_root" -mindepth 2 -maxdepth 2 -name snapshot.json -type f | sort)
  if (( found < 2 )); then
    echo "ERROR: production capture supplied fewer than two frozen states" >&2
    exit 2
  fi
}

build_family_from_roots() {
  local snapshot_root=$1
  local load_root=$2
  local outroot=$3
  local family_out=$4
  local iteration_mode=$5
  local args=(
    --snapshot-root "$snapshot_root"
    --load-invariance-root "$load_root"
    --mechanical-config "$CONFIG"
    --outroot "$outroot"
    --family-out "$family_out"
    --target-extension-um "$TARGET_EXT_UM"
    --theta-deg "$THETA"
    --da-phys-um "$DA_PHYS_UM"
    --event-minimum-factor "$EVENT_MINIMUM_FACTOR"
    --event-maximum-factor "$EVENT_MAXIMUM_FACTOR"
    --margin-events "$MARGIN_EVENTS"
  )
  if [[ "$iteration_mode" == iteration ]]; then
    args+=(--allow-unconverged-capture)
  fi
  "$PYTHON_BIN" scripts/build_v10_2_27_kernel_from_current_mechanics.py "${args[@]}"
}

SNAPSHOT_ROOT=${KERNEL_SNAPSHOT_ROOT:-}
SNAPSHOT_ARCHIVE=${KERNEL_SNAPSHOT_ARCHIVE:-}
LOAD_ROOT=${KERNEL_LOAD_INVARIANCE_ROOT:-}
LOAD_ARCHIVE=${KERNEL_LOAD_INVARIANCE_ARCHIVE:-}

if [[ -n "$SNAPSHOT_ARCHIVE" || -n "$LOAD_ARCHIVE" ]]; then
  if [[ -z "$SNAPSHOT_ARCHIVE" || -z "$LOAD_ARCHIVE" ]]; then
    echo "ERROR: snapshot and load-invariance archives must be supplied together" >&2
    exit 2
  fi
  exec "$PYTHON_BIN" scripts/build_v10_2_27_kernel_from_current_mechanics.py \
    --snapshot-archive "$SNAPSHOT_ARCHIVE" \
    --load-invariance-archive "$LOAD_ARCHIVE" \
    --mechanical-config "$CONFIG" \
    --outroot "$CACHE_DIR" \
    --family-out "$FAMILY_OUT" \
    --target-extension-um "$TARGET_EXT_UM" \
    --theta-deg "$THETA" \
    --da-phys-um "$DA_PHYS_UM" \
    --event-minimum-factor "$EVENT_MINIMUM_FACTOR" \
    --event-maximum-factor "$EVENT_MAXIMUM_FACTOR" \
    --margin-events "$MARGIN_EVENTS"
fi

if [[ -n "$SNAPSHOT_ROOT" ]]; then
  capture_audits "$SNAPSHOT_ROOT"
  if [[ -z "$LOAD_ROOT" ]]; then
    LOAD_ROOT="$CACHE_DIR/load_invariance"
    build_load_invariance "$SNAPSHOT_ROOT" "$LOAD_ROOT"
  fi
  exec "$PYTHON_BIN" scripts/build_v10_2_27_kernel_from_current_mechanics.py \
    --snapshot-root "$SNAPSHOT_ROOT" \
    --load-invariance-root "$LOAD_ROOT" \
    --mechanical-config "$CONFIG" \
    --outroot "$CACHE_DIR" \
    --family-out "$FAMILY_OUT" \
    --target-extension-um "$TARGET_EXT_UM" \
    --theta-deg "$THETA" \
    --da-phys-um "$DA_PHYS_UM" \
    --event-minimum-factor "$EVENT_MINIMUM_FACTOR" \
    --event-maximum-factor "$EVENT_MAXIMUM_FACTOR" \
    --margin-events "$MARGIN_EVENTS"
fi

if [[ -z "${KERNEL_CAPTURE_SEED_FAMILY:-}" ]]; then
  echo "ERROR: automatic mechanics-only kernel capture is disabled." >&2
  echo "Fresh calculation requires KERNEL_CAPTURE_SEED_FAMILY, an explicit" >&2
  echo "accepted production bootstrap family, plus the production capture option" >&2
  echo "and hazard seed. The bootstrap operator is replaced iteratively and cannot" >&2
  echo "be promoted until consecutive target-grid families converge." >&2
  exit 5
fi
CAPTURE_COMMAND=${KERNEL_CAPTURE_COMMAND:-"bash scripts/run_v10_2_27_accepted_production_kernel_capture.sh"}
ORIGINAL_SEED=$(cd "$(dirname "$KERNEL_CAPTURE_SEED_FAMILY")" && pwd)/$(basename "$KERNEL_CAPTURE_SEED_FAMILY")
[[ -f "$ORIGINAL_SEED" ]] || { echo "ERROR: missing bootstrap family: $ORIGINAL_SEED" >&2; exit 2; }
ORIGINAL_SEED_SHA=$(sha256_file "$ORIGINAL_SEED")
if [[ -n "${KERNEL_CAPTURE_SEED_FAMILY_EXPECTED_SHA256:-}" && \
      "$ORIGINAL_SEED_SHA" != "$KERNEL_CAPTURE_SEED_FAMILY_EXPECTED_SHA256" ]]; then
  echo "ERROR: initial bootstrap family SHA-256 mismatch" >&2
  exit 2
fi

SELF_ROOT="$CACHE_DIR/self_consistency"
rm -rf "$SELF_ROOT"
mkdir -p "$SELF_ROOT"
CURRENT_SEED="$ORIGINAL_SEED"
CURRENT_SEED_SHA="$ORIGINAL_SEED_SHA"
PREVIOUS_TARGET_FAMILY=""
CONVERGED=0
FINAL_ITERATION=""
FINAL_SNAPSHOT_ROOT=""
FINAL_LOAD_ROOT=""
FINAL_CANDIDATE=""
COMPARISON_FILES=()

for (( iteration=0; iteration<MAX_ITERATIONS; iteration++ )); do
  ITER_ROOT=$(printf '%s/iteration_%02d' "$SELF_ROOT" "$iteration")
  ITER_SNAPSHOTS="$ITER_ROOT/snapshots"
  ITER_LOADS="$ITER_ROOT/load_invariance"
  ITER_BUILD="$ITER_ROOT/build"
  ITER_FAMILY="$ITER_ROOT/family.json"
  mkdir -p "$ITER_ROOT"

  export KERNEL_CAPTURE_SEED_FAMILY="$CURRENT_SEED"
  export KERNEL_CAPTURE_SEED_FAMILY_EXPECTED_SHA256="$CURRENT_SEED_SHA"
  export V10227_KERNEL_CAPTURE_OUTROOT="$ITER_SNAPSHOTS"
  export V10227_KERNEL_CAPTURE_THETA_DEG="$THETA"
  export V10227_KERNEL_CAPTURE_TARGET_EXTENSION_UM="$TARGET_EXT_UM"
  export V10227_KERNEL_CAPTURE_REQUIRED_MAX_EXTENSION_UM="$REQUIRED_EXT_UM"
  export V10227_KERNEL_CAPTURE_TEMPERATURE_K="$CAPTURE_TEMPERATURE_K"
  export V10227_KERNEL_SELF_CONSISTENCY_ITERATION="$iteration"

  echo "SELF-CONSISTENCY iteration $iteration: capture with seed $CURRENT_SEED_SHA" >&2
  rm -rf "$ITER_SNAPSHOTS"
  mkdir -p "$(dirname "$ITER_SNAPSHOTS")"
  bash -lc "$CAPTURE_COMMAND"
  for required in \
    "$ITER_SNAPSHOTS/capture_complete.json" \
    "$ITER_SNAPSHOTS/kernel_capture_manifest.json"; do
    [[ -f "$required" ]] || { echo "ERROR: capture did not create $required" >&2; exit 2; }
  done
  capture_audits "$ITER_SNAPSHOTS"
  build_load_invariance "$ITER_SNAPSHOTS" "$ITER_LOADS"
  rm -rf "$ITER_BUILD"
  mkdir -p "$ITER_BUILD"
  build_family_from_roots \
    "$ITER_SNAPSHOTS" "$ITER_LOADS" "$ITER_BUILD" "$ITER_FAMILY" iteration

  if [[ -n "$PREVIOUS_TARGET_FAMILY" ]]; then
    COMPARISON="$ITER_ROOT/self_consistency_comparison.json"
    set +e
    "$PYTHON_BIN" scripts/compare_v10_2_27_kernel_families.py \
      --previous "$PREVIOUS_TARGET_FAMILY" \
      --current "$ITER_FAMILY" \
      --maximum-relative-kernel-change "$MAX_RELATIVE_CHANGE" \
      --maximum-absolute-kernel-change-Pa-sqrt-m-per-line "$MAX_ABSOLUTE_CHANGE" \
      --maximum-extension-change-um "$MAX_EXTENSION_CHANGE_UM" \
      --maximum-normalization-relative-change "$MAX_NORMALIZATION_CHANGE" \
      --output "$COMPARISON"
    comparison_status=$?
    set -e
    COMPARISON_FILES+=("$COMPARISON")
    if (( comparison_status == 0 )); then
      CONVERGED=1
      FINAL_ITERATION="$iteration"
      FINAL_SNAPSHOT_ROOT="$ITER_SNAPSHOTS"
      FINAL_LOAD_ROOT="$ITER_LOADS"
      FINAL_CANDIDATE="$ITER_FAMILY"
      echo "SELF-CONSISTENCY converged at iteration $iteration" >&2
      break
    fi
    if (( comparison_status != 3 )); then
      echo "ERROR: kernel family comparison failed with status $comparison_status" >&2
      exit "$comparison_status"
    fi
  fi

  PREVIOUS_TARGET_FAMILY="$ITER_FAMILY"
  CURRENT_SEED="$ITER_FAMILY"
  CURRENT_SEED_SHA=$(sha256_file "$CURRENT_SEED")
done

if (( CONVERGED != 1 )); then
  echo "ERROR: target kernel did not converge within $MAX_ITERATIONS iterations" >&2
  exit 6
fi

FINAL_SELECTION="$FINAL_SNAPSHOT_ROOT/kernel_self_consistency_selection.json"
"$PYTHON_BIN" - \
  "$FINAL_SELECTION" "$CONFIG" "$ORIGINAL_SEED" "$ORIGINAL_SEED_SHA" \
  "$FINAL_ITERATION" "$FINAL_CANDIDATE" "${COMPARISON_FILES[@]}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
selection, config, original, original_sha, iteration, candidate, *comparisons = sys.argv[1:]
candidate_path = Path(candidate).resolve()
payload = {
    "schema": "v10.2.27_kernel_self_consistency_selection_v1",
    "mechanical_configuration": str(Path(config).resolve()),
    "initial_bootstrap_family": str(Path(original).resolve()),
    "initial_bootstrap_family_sha256": original_sha,
    "converged": True,
    "converged_iteration": int(iteration),
    "minimum_target_family_passes": 2,
    "converged_candidate_family": str(candidate_path),
    "converged_candidate_family_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
    "comparison_audits": [str(Path(path).resolve()) for path in comparisons],
}
Path(selection).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

rm -f \
  "$FAMILY_OUT" \
  "$CACHE_DIR/mechanics_normalization.json" \
  "$CACHE_DIR/coverage_audit.json" \
  "$CACHE_DIR/kernel_build_manifest.json" \
  "$CACHE_DIR/kernel_self_consistency_manifest.json"
rm -rf "$CACHE_DIR/portable_load_invariance_reports"

build_family_from_roots \
  "$FINAL_SNAPSHOT_ROOT" "$FINAL_LOAD_ROOT" "$CACHE_DIR" "$FAMILY_OUT" final

FINAL_SHA=$(sha256_file "$FAMILY_OUT")
CANDIDATE_SHA=$(sha256_file "$FINAL_CANDIDATE")
if [[ "$FINAL_SHA" != "$CANDIDATE_SHA" ]]; then
  echo "ERROR: canonical rebuild differs from converged candidate" >&2
  echo "Candidate: $CANDIDATE_SHA" >&2
  echo "Canonical: $FINAL_SHA" >&2
  exit 2
fi

"$PYTHON_BIN" - \
  "$CACHE_DIR/kernel_self_consistency_manifest.json" \
  "$FINAL_SELECTION" "$FAMILY_OUT" "$FINAL_SHA" "$FINAL_ITERATION" \
  "$MAX_ITERATIONS" "$MAX_RELATIVE_CHANGE" "$MAX_ABSOLUTE_CHANGE" \
  "$MAX_EXTENSION_CHANGE_UM" "$MAX_NORMALIZATION_CHANGE" <<'PY'
import json
from pathlib import Path
import sys
(
    output,
    selection,
    family,
    family_sha,
    converged_iteration,
    max_iterations,
    max_relative,
    max_absolute,
    max_extension,
    max_normalization,
) = sys.argv[1:]
selection_payload = json.loads(Path(selection).read_text())
payload = {
    "schema": "v10.2.27_kernel_self_consistency_manifest_v1",
    "converged": True,
    "converged_iteration": int(converged_iteration),
    "maximum_iterations": int(max_iterations),
    "canonical_family": str(Path(family).resolve()),
    "canonical_family_sha256": family_sha,
    "selection": selection_payload,
    "tolerances": {
        "maximum_relative_kernel_change": float(max_relative),
        "maximum_absolute_kernel_change_Pa_sqrt_m_per_line": float(max_absolute),
        "maximum_extension_change_um": float(max_extension),
        "maximum_normalization_relative_change": float(max_normalization),
    },
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

printf 'Self-consistent v10.2.27 kernel complete\n'
printf '  converged iteration: %s\n' "$FINAL_ITERATION"
printf '  family: %s\n' "$FAMILY_OUT"
printf '  family SHA256: %s\n' "$FINAL_SHA"
