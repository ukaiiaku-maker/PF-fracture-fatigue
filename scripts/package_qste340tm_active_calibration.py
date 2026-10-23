#!/usr/bin/env python3
"""Build a source-complete QSTE340TM active-calibration evidence archive."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.complete_qste340tm_active_calibration import OUT
from scripts.refit_qste340tm_restored_native import ALT_REGISTRY, DATA, PARENT_REGISTRY

ARCHIVE = ROOT / "QSTE340TM_ACTIVE_RESTORED_CALIBRATION.zip"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def add_tree(entries: dict[str, Path], source: Path, prefix: str, pattern: str = "*") -> None:
    for path in sorted(source.rglob(pattern)):
        if path.is_file() and "__pycache__" not in path.parts:
            entries[f"{prefix}/{path.relative_to(source).as_posix()}"] = path


def main() -> int:
    entries: dict[str, Path] = {}
    add_tree(entries, OUT, "results")
    entries.pop("results/PACKAGE_MANIFEST.json", None)
    add_tree(entries, ROOT/"arrhenius_fracture", "source/arrhenius_fracture", "*.py")
    for relative in (
        "scripts/complete_qste340tm_active_calibration.py",
        "scripts/analyze_qste340tm_active_calibration.py",
        "scripts/package_qste340tm_active_calibration.py",
        "scripts/verify_qste340tm_active_package.py",
        "scripts/refit_qste340tm_restored_native.py",
        "scripts/fatigue_informed_joint_barrier_v3_v10230.py",
        "scripts/thermodynamic_joint_barrier_v10230.py",
        "tests/test_qste340tm_active_completion.py",
        "tests/test_qste340tm_restored_binding.py",
        "integration_evidence/r4_same_engine_coupled_spatial/dependencies/signed_kernel_family.json",
        "integration_evidence/r10_alternative_gstar_joint_search/alternative_gstar_registry.json",
    ):
        entries[f"source/{relative}"] = ROOT/relative
    add_tree(entries, DATA/"normalized", "inputs/normalized")
    for name in ("verify_import.py", "README.md", "IMPORT_MANIFEST.json"):
        path = DATA/name
        if path.is_file():
            entries[f"inputs/{name}"] = path
    entries["registries/v3_complete_candidate_registry.json"] = Path(PARENT_REGISTRY)
    entries["registries/alternative_gstar_registry.json"] = Path(ALT_REGISTRY)
    missing = [name for name, path in entries.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"package inputs missing: {missing}")
    manifest = {
        "schema": "qste340tm-active-calibration-package/1",
        "root_commit": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        "files": [
            {"archive_path": name, "source_path": str(path.resolve()),
             "size_bytes": path.stat().st_size, "sha256": digest(path)}
            for name, path in sorted(entries.items())
        ],
    }
    manifest_path = OUT/"PACKAGE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    entries["PACKAGE_MANIFEST.json"] = manifest_path
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as archive:
        for name, path in sorted(entries.items()):
            archive.write(path, name)
    print(json.dumps({
        "archive": str(ARCHIVE), "sha256": digest(ARCHIVE),
        "size_bytes": ARCHIVE.stat().st_size, "files": len(entries),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
