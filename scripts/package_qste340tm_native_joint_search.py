#!/usr/bin/env python3
"""Bind the executed QSTE340TM search to inspectable source and replay inputs."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.qualify_qste340tm_native_transfer import OUT


SOURCE_FILES = (
    "arrhenius_fracture/unified_mpz.py",
    "arrhenius_fracture/continuum_source_tip.py",
    "arrhenius_fracture/material_manifest.py",
    "scripts/calibrate_qste340tm_stateful_exp_floor.py",
    "scripts/qste340tm_faithful_reduction.py",
    "scripts/refit_qste340tm_faithful_reduction.py",
    "scripts/fatigue_informed_joint_barrier_v3_v10230.py",
    "scripts/qualify_qste340tm_native_transfer.py",
    "scripts/continue_qste340tm_native_joint_search.py",
    "scripts/finalize_qste340tm_joint_search.py",
    "scripts/package_qste340tm_native_joint_search.py",
    "tests/test_qste340tm_faithful_reduction.py",
    "tests/test_qste340tm_native_joint_search.py",
)

INPUT_FILES = (
    "literature_data/fatigue_recovered_review/normalized/qste340tm_constant_amplitude.csv",
    "literature_data/fatigue_recovered_review/model_conditional/stateful_exp_floor_capacity_policy/faithful_retained_reduction/fit_faithful_reduction.json",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def main() -> int:
    snapshot = OUT / "source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    source_records = []
    for relative in SOURCE_FILES:
        source = ROOT / relative
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_records.append({"path": relative, "sha256": digest(source), "bytes": source.stat().st_size})

    input_records = []
    for relative in INPUT_FILES:
        source = ROOT / relative
        input_records.append({"path": relative, "sha256": digest(source), "bytes": source.stat().st_size})

    output_records = []
    for name in (
        "joint_search.json",
        "joint_predictions.csv",
        "fixed_Kmax_contrasts.csv",
        "native_transfer_correspondence.json",
        "joint_fit_DeltaK.png",
        "joint_fit_Kmax.png",
        "README.md",
    ):
        path = OUT / name
        output_records.append({"path": name, "sha256": digest(path), "bytes": path.stat().st_size})

    payload = {
        "schema": "qste340tm-native-joint-search-execution-manifest/1",
        "repository": {
            "branch": git("branch", "--show-current"),
            "starting_commit": "3c840232",
            "starting_commit_full": git("rev-parse", "3c840232"),
            "worktree_head_when_packaged": git("rev-parse", "HEAD"),
            "dirty_source_snapshot": True,
        },
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "matplotlib", "pytest")
            },
        },
        "fixed_conditions": {
            "temperature_K": 298.15,
            "temperature_C": 25.0,
            "frequency_Hz": 10.0,
            "frequency_status": "provisional assumption",
            "reported_DeltaK_convention": "DeltaK; Kmax = DeltaK/(1-R)",
            "material_count": 1,
            "R_values": [0.1, 0.3, 0.5, 0.7],
            "empirical_retained_capacity": None,
            "endpoint_survivor_as_retained_source": False,
        },
        "replay": [
            "python verify_import.py",
            "python scripts/qualify_qste340tm_native_transfer.py --panels 32 128 512 2048",
            "python scripts/continue_qste340tm_native_joint_search.py --iterations 6 --substeps 1 --waveform 32",
            "python scripts/finalize_qste340tm_joint_search.py",
            "python scripts/package_qste340tm_native_joint_search.py",
            "python -m pytest -q tests/test_qste340tm_faithful_reduction.py tests/test_qste340tm_native_joint_search.py",
        ],
        "source_files": source_records,
        "input_files": input_records,
        "output_files": output_records,
        "integrity_note": "execution_manifest.json omits its own hash to avoid a circular digest",
    }
    (OUT / "execution_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"source_files": len(source_records), "inputs": len(input_records), "outputs": len(output_records), "manifest": str(OUT / "execution_manifest.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
