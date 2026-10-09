#!/usr/bin/env python3
"""Repair only missing-input operations in an otherwise valid recovery shard.

The original recovery shard completed both Kirsch executions but its sparse
checkout omitted retained manifests needed by the patch and transfer audits.
This tool verifies that exact failure identity, retains the successful outputs,
and independently reruns only the two operations that never started science.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

FAILED_OPERATIONS = {
    "patch_operator": (
        "qualify_cavity_boundary_patch_recovery_v1.py",
        "operator.json",
        "traction/sha256_manifest.json",
    ),
    "recovery_transfer": (
        "qualify_cavity_recovery_transfer_budget_v1.py",
        "transfer.json",
        "production/sha256_manifest.json",
    ),
}
RETAINED_OPERATIONS = ("kirsch_coarse", "kirsch_fine")


def _read(path: Path):
    return json.loads(path.read_text())


def _inventory(root: Path):
    manifest = _read(root / "sha256_manifest.json")
    actual = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "sha256_manifest.json"
    }
    if manifest != actual:
        raise ValueError(f"invalid source recovery inventory: {root}")
    return actual


def _git(root: Path, *args: str):
    return subprocess.check_output(("git", *args), cwd=root, text=True).strip()


def require_missing_input_failure(original: dict, implementation_sha: str, failure_logs: dict[str, str]):
    expected = {
        "schema": "v5.source-resolution-final-phase-execution/1",
        "phase": "recovery", "section": "all", "shard_index": 0,
        "shard_count": 1, "executed_code_sha": implementation_sha,
        "execution_completed": False, "clean_exact_head_at_end": True,
    }
    if any(original.get(key) != value for key, value in expected.items()):
        raise ValueError("unexpected failed recovery execution identity")
    operations = {row["operation"]: row for row in original["operations"]}
    if set(operations) != set(FAILED_OPERATIONS) | set(RETAINED_OPERATIONS):
        raise ValueError("unexpected recovery operation set")
    if any(operations[name]["returncode"] != 0 for name in RETAINED_OPERATIONS):
        raise ValueError("a retained recovery operation did not succeed")
    if any(operations[name]["returncode"] != 1 for name in FAILED_OPERATIONS):
        raise ValueError("unexpected missing-input operation result")
    for name, (_, _, missing_suffix) in FAILED_OPERATIONS.items():
        failure_log = failure_logs[name]
        if "FileNotFoundError" not in failure_log or missing_suffix not in failure_log:
            raise ValueError(f"{name} was not the authorized missing-input failure")
    return operations


def repair(source: Path, output: Path, implementation_root: Path, *, require_pinned: bool):
    if output.exists():
        raise ValueError("refusing to overwrite recovery repair")
    implementation_root = implementation_root.resolve()
    implementation_sha = _git(implementation_root, "rev-parse", "HEAD")
    if _git(implementation_root, "status", "--porcelain"):
        raise ValueError("exact implementation checkout is not clean")
    from v5_numerical_runtime_v1 import require_pinned as require_runtime, runtime_record
    kernels = require_runtime(runtime_record()) if require_pinned else None

    source_inventories = {side: _inventory(source / side) for side in ("a", "b")}
    if source_inventories["a"] != source_inventories["b"]:
        raise ValueError("failed recovery source A/B inventories are not exact")

    for side in ("a", "b"):
        original = _read(source / side / "execution.json")
        failure_logs = {name: (source / side / f"{name}.log").read_text() for name in FAILED_OPERATIONS}
        require_missing_input_failure(original, implementation_sha, failure_logs)
        if kernels is not None and original["selected_numerical_kernels"] != kernels:
            raise ValueError("repair numerical runtime differs from source shard")

        destination = output / side
        destination.mkdir(parents=True)
        for name in RETAINED_OPERATIONS:
            shutil.copytree(source / side / name, destination / name)
            shutil.copy2(source / side / f"{name}.log", destination / f"{name}.log")
        for name, (script, filename, _) in FAILED_OPERATIONS.items():
            command = [sys.executable, str(implementation_root / "scripts" / script), str(destination / filename)]
            with (destination / f"{name}.log").open("w") as log:
                result = subprocess.run(command, cwd=implementation_root, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise ValueError(f"repaired {name} operation failed with {result.returncode}")

        execution = dict(original)
        execution["operations"] = [dict(row, returncode=0) for row in original["operations"]]
        execution["execution_completed"] = True
        execution["repair_provenance"] = {
            "classification": "MISSING_SPARSE_CHECKOUT_INPUTS_ONLY",
            "retained_successful_operations": list(RETAINED_OPERATIONS),
            "independently_rerun_operations": list(FAILED_OPERATIONS),
            "source_inventory_sha256": hashlib.sha256(
                json.dumps(source_inventories[side], sort_keys=True).encode()
            ).hexdigest(),
            "scientific_predicates_or_tolerances_changed": False,
        }
        (destination / "execution.json").write_text(json.dumps(execution, sort_keys=True, indent=2) + "\n")
        manifest = {
            str(path.relative_to(destination)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(destination.rglob("*"))
            if path.is_file() and path.name != "sha256_manifest.json"
        }
        (destination / "sha256_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")

    repaired = {side: _inventory(output / side) for side in ("a", "b")}
    if repaired["a"] != repaired["b"]:
        raise ValueError("repaired recovery A/B inventories are not exact")
    if _git(implementation_root, "status", "--porcelain"):
        raise ValueError("repair mutated the exact implementation checkout")
    print(json.dumps({
        "classification": "RECOVERY_SHARD_MISSING_INPUT_REPAIR_COMPLETE",
        "implementation_sha": implementation_sha,
        "exact_recursive_comparison": True,
        "file_count_per_side": len(repaired["a"]),
    }, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--implementation-root", type=Path, required=True)
    parser.add_argument("--require-pinned-runtime", action="store_true")
    args = parser.parse_args()
    repair(args.source, args.output, args.implementation_root, require_pinned=args.require_pinned_runtime)


if __name__ == "__main__":
    main()
