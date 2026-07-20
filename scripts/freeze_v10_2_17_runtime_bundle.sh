#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

OUTROOT=${OUTROOT:?Set OUTROOT to the v10.2.17 campaign output directory}
SOURCE_FAMILY=${FAMILY_JSON:-$OUTROOT/mechanics/v10_2_14_active_only_campaign_family.json}
BUNDLE_DIR=${BUNDLE_DIR:-$ROOT/runtime_inputs/v10_2_17}
DEST_FAMILY="$BUNDLE_DIR/v10_2_14_active_only_campaign_family.json"
ENGINE_CONFIG_SOURCE=${ENGINE_CONFIG_SOURCE:-${ENGINE_CONFIG:-}}

if [[ ! -s "$SOURCE_FAMILY" ]]; then
  echo "ERROR: source family is missing or empty: $SOURCE_FAMILY" >&2
  exit 2
fi

mkdir -p "$BUNDLE_DIR"
if [[ "$(cd "$(dirname "$SOURCE_FAMILY")" && pwd)/$(basename "$SOURCE_FAMILY")" != \
      "$(cd "$BUNDLE_DIR" && pwd)/$(basename "$DEST_FAMILY")" ]]; then
  cp -p "$SOURCE_FAMILY" "$DEST_FAMILY"
fi

if [[ -n "$ENGINE_CONFIG_SOURCE" && -f "$ENGINE_CONFIG_SOURCE" ]]; then
  cp -p "$ENGINE_CONFIG_SOURCE" "$BUNDLE_DIR/v10_2_3_2d_engine_config.json"
fi

python --version > "$BUNDLE_DIR/python_version.txt" 2>&1
python -m pip freeze > "$BUNDLE_DIR/pip_freeze.txt"
if command -v conda >/dev/null 2>&1; then
  conda env export --from-history > "$BUNDLE_DIR/conda_environment_from_history.yml" || true
fi
git rev-parse HEAD > "$BUNDLE_DIR/git_commit.txt"
git status --short > "$BUNDLE_DIR/git_status.txt"

BUNDLE_DIR="$BUNDLE_DIR" DEST_FAMILY="$DEST_FAMILY" ROOT="$ROOT" python - <<'PY'
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import arrhenius_fracture
from arrhenius_fracture.signed_kernel_family_v10214 import (
    ActiveOnlySigned2DShieldingKernelFamily,
)

root = Path(os.environ["ROOT"]).resolve()
bundle = Path(os.environ["BUNDLE_DIR"]).resolve()
family_path = Path(os.environ["DEST_FAMILY"]).resolve()
required = [
    root / "arrhenius_fracture" / "sharp_front_v10_2_17.py",
    root / "arrhenius_fracture" / "sharp_front_v10_1_7_5.py",
    root / "arrhenius_fracture" / "state_resolved_signed_engine_v10214.py",
    root / "arrhenius_fracture" / "signed_kernel_family_v10214.py",
    root / "arrhenius_fracture" / "parameter_registry_v9111.py",
    root / "arrhenius_fracture" / "data" / "materials" / "MPZ_v9_11_1_parameter_registry.csv",
    root / "scripts" / "run_v10_2_17_stage3_overnight.sh",
    root / "scripts" / "run_v10_2_17_stage3_monotonic_temperature_sweep.sh",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit("missing required code files:\n" + "\n".join(missing))
package = Path(arrhenius_fracture.__file__).resolve().parent
if package != root / "arrhenius_fracture":
    raise SystemExit(f"stale editable import: expected {root / 'arrhenius_fracture'}, got {package}")
family = ActiveOnlySigned2DShieldingKernelFamily.from_json(family_path)
if family.metadata.get("production_parameterization_allowed") is not True:
    raise SystemExit("frozen family is not production-enabled")

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

artifact_names = [
    "v10_2_14_active_only_campaign_family.json",
    "v10_2_3_2d_engine_config.json",
    "python_version.txt",
    "pip_freeze.txt",
    "conda_environment_from_history.yml",
    "git_commit.txt",
    "git_status.txt",
]
files = [bundle / name for name in artifact_names if (bundle / name).is_file()]
payload = {
    "schema": "v10.2.17_self_contained_runtime_bundle",
    "repository_root": str(root),
    "local_package": str(package),
    "family_path": str(family_path),
    "family_states": len(family.states),
    "production_parameterization_allowed": True,
    "external_legacy_install_required_to_resume": False,
    "external_mechanics_inputs_required_only_to_rebuild_family_from_raw_responses": True,
    "engine_config_preserved_for_provenance": (bundle / "v10_2_3_2d_engine_config.json").is_file(),
    "required_code_files": [str(path.relative_to(root)) for path in required],
    "artifacts": [
        {"path": str(path.relative_to(root)), "sha256": digest(path), "bytes": path.stat().st_size}
        for path in files
    ],
}
manifest = bundle / "runtime_bundle_manifest.json"
manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
print(f"runtime bundle ready: {manifest}")
PY
