#!/usr/bin/env python3
"""Create the lightweight immutable source snapshot for the managed worker."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = Path(os.environ.get(
    "NATURAL_VOID_SNAPSHOT_DEST",
    ROOT / "runs/managed_natural_void_v1_20260919/source",
))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    if DESTINATION.exists():
        raise RuntimeError(f"immutable snapshot already exists: {DESTINATION}")
    DESTINATION.mkdir(parents=True)
    shutil.copytree(
        ROOT / "arrhenius_fracture", DESTINATION / "arrhenius_fracture",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (DESTINATION / "scripts").mkdir()
    required_scripts = (
        "run_natural_void_hypothesis_v1.py",
        "run_sustained_growth_candidate_v1.py",
        "prepare_sustained_growth_candidate_v1.py",
        "run_managed_sustained_growth_loop.sh",
        "run_active_two_mechanism_natural_void_study.py",
        "screen_informative_off_axis_operating_point.py",
    )
    for name in required_scripts:
        shutil.copy2(ROOT / "scripts" / name, DESTINATION / "scripts" / name)
    scout_dir = DESTINATION / "artifacts/natural_void_hypothesis_scout_v1"
    scout_dir.mkdir(parents=True)
    shutil.copy2(
        ROOT / "artifacts/natural_void_hypothesis_scout_v1/scout.json",
        scout_dir / "scout.json",
    )
    for optional in ("pyproject.toml", "pytest.ini"):
        if (ROOT / optional).exists():
            shutil.copy2(ROOT / optional, DESTINATION / optional)
    files = sorted(path for path in DESTINATION.rglob("*") if path.is_file())
    manifest = {
        "schema": "managed-natural-void-source-snapshot/1",
        "source_root": str(ROOT),
        "snapshot_root": str(DESTINATION),
        "file_count": len(files),
        "files": {
            path.relative_to(DESTINATION).as_posix(): sha256(path) for path in files
        },
        "includes_uncommitted_and_untracked_runtime_source": True,
        "historical_artifact_tree_copied": False,
    }
    (DESTINATION / "snapshot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    for path in DESTINATION.rglob("*"):
        mode = path.stat().st_mode
        path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    DESTINATION.chmod(DESTINATION.stat().st_mode & ~stat.S_IWUSR)
    print(DESTINATION)
    print("SNAPSHOT_FILES", len(files))


if __name__ == "__main__":
    main()
