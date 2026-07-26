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

THETAS=${THETAS:-"0 18 30"}
TARGETS_UM=${TARGETS_UM:-"50"}
OUTROOT=${OUTROOT:-runs/v10_2_28_direct_kernel_orientation_coverage_smoke_v1}
CACHE_ROOT=${CACHE_ROOT:-runs/v10_2_28_kernel_cache}
HEARTBEAT_SECONDS=${HEARTBEAT_SECONDS:-60}

case "$HEARTBEAT_SECONDS" in
  ''|*[!0-9]*) echo "ERROR: HEARTBEAT_SECONDS must be a positive integer" >&2; exit 2 ;;
esac
[[ "$HEARTBEAT_SECONDS" -ge 1 ]] || {
  echo "ERROR: HEARTBEAT_SECONDS must be at least one" >&2
  exit 2
}

mkdir -p "$OUTROOT"

manifest="$OUTROOT/smoke_results.jsonl"
: > "$manifest"

run_logged() {
  local label=$1
  local stdout_path=$2
  local stderr_path=$3
  shift 3

  local start_epoch
  start_epoch=$(date +%s)
  echo "CASE_START: ${label} pid_pending"
  "$@" > "$stdout_path" 2> "$stderr_path" &
  local pid=$!
  echo "CASE_PID: ${label} pid=${pid}"

  while kill -0 "$pid" 2>/dev/null; do
    sleep "$HEARTBEAT_SECONDS"
    if kill -0 "$pid" 2>/dev/null; then
      local now elapsed
      now=$(date +%s)
      elapsed=$((now - start_epoch))
      echo "HEARTBEAT: ${label} pid=${pid} elapsed_s=${elapsed}"
    fi
  done

  local rc=0
  wait "$pid" || rc=$?
  local end_epoch elapsed
  end_epoch=$(date +%s)
  elapsed=$((end_epoch - start_epoch))
  if [[ "$rc" -eq 0 ]]; then
    echo "CASE_COMPLETE: ${label} elapsed_s=${elapsed}"
  else
    echo "CASE_FAILED: ${label} exit=${rc} elapsed_s=${elapsed}" >&2
    tail -n 100 "$stderr_path" >&2 || true
  fi
  return "$rc"
}

for theta in $THETAS; do
  for target in $TARGETS_UM; do
    tag="theta${theta}_target${target}um"
    build_json="$OUTROOT/${tag}_build.json"
    reuse_json="$OUTROOT/${tag}_reuse.json"
    build_log="$OUTROOT/${tag}_build.log"
    reuse_log="$OUTROOT/${tag}_reuse.log"

    run_logged \
      "build:${tag}" \
      "$build_json" \
      "$build_log" \
      "$PYTHON_BIN" scripts/ensure_v10_2_28_signed_kernel.py \
        --theta-deg "$theta" \
        --target-extension-um "$target" \
        --branching-mode single_front \
        --maximum-fronts 1 \
        --mode auto \
        --cache-root "$CACHE_ROOT" \
        --json

    run_logged \
      "reuse:${tag}" \
      "$reuse_json" \
      "$reuse_log" \
      "$PYTHON_BIN" scripts/ensure_v10_2_28_signed_kernel.py \
        --theta-deg "$theta" \
        --target-extension-um "$target" \
        --branching-mode single_front \
        --maximum-fronts 1 \
        --mode reuse-only \
        --cache-root "$CACHE_ROOT" \
        --json

    BUILD_JSON="$build_json" REUSE_JSON="$reuse_json" \
    THETA="$theta" TARGET_UM="$target" "$PYTHON_BIN" - <<'PY' >> "$manifest"
import json
import os
from pathlib import Path

build = json.loads(Path(os.environ["BUILD_JSON"]).read_text())
reuse = json.loads(Path(os.environ["REUSE_JSON"]).read_text())
if reuse.get("resolution") not in {"local_cache", "local_registry", "tracked_registry"}:
    raise SystemExit(f"second resolution did not reuse a validated family: {reuse}")
for key in ("configuration_fingerprint", "file_sha256", "physics_fingerprint", "family"):
    if build.get(key) != reuse.get(key):
        raise SystemExit(f"cache-reuse mismatch for {key}: {build.get(key)} != {reuse.get(key)}")
print(json.dumps({
    "theta_deg": float(os.environ["THETA"]),
    "target_extension_um": float(os.environ["TARGET_UM"]),
    "first_resolution": build.get("resolution"),
    "second_resolution": reuse.get("resolution"),
    "configuration_fingerprint": build["configuration_fingerprint"],
    "family": build["family"],
    "family_sha256": build["file_sha256"],
    "physics_fingerprint": build["physics_fingerprint"],
    "maximum_extension_um": build["maximum_extension_um"],
    "cache_reuse_verified": True,
}, sort_keys=True))
PY
  done
done

OUTROOT="$OUTROOT" THETAS="$THETAS" TARGETS_UM="$TARGETS_UM" \
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUTROOT"]).resolve()
rows = [
    json.loads(line)
    for line in (root / "smoke_results.jsonl").read_text().splitlines()
    if line.strip()
]
expected = len(os.environ["THETAS"].split()) * len(os.environ["TARGETS_UM"].split())
if len(rows) != expected or not all(row.get("cache_reuse_verified") is True for row in rows):
    raise SystemExit("direct-kernel smoke matrix is incomplete")
fingerprints = {(row["theta_deg"], row["configuration_fingerprint"]) for row in rows}
by_theta = {}
for theta, fingerprint in fingerprints:
    by_theta.setdefault(theta, set()).add(fingerprint)
unique_by_theta = {next(iter(values)) for values in by_theta.values()}
if len(by_theta) > 1 and len(unique_by_theta) != len(by_theta):
    raise SystemExit("different orientations unexpectedly produced the same mechanical fingerprint")
comparison_applicable = len(by_theta) > 1
payload = {
    "schema": "v10.2.28_direct_kernel_orientation_coverage_smoke_v1",
    "passed": True,
    "case_count": len(rows),
    "theta_values_deg": sorted(by_theta),
    "target_extensions_um": sorted({row["target_extension_um"] for row in rows}),
    "all_cache_reuse_verified": True,
    "orientation_fingerprint_comparison_applicable": comparison_applicable,
    "different_orientation_fingerprints_verified": (
        comparison_applicable and len(unique_by_theta) == len(by_theta)
    ),
    "records": rows,
}
(root / "smoke_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "CAMPAIGN_COMPLETE: direct-kernel smoke output=$OUTROOT"
