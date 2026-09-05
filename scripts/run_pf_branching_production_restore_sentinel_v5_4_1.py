#!/usr/bin/env python3
"""Exercise the actual production restart entrypoint and stop before lookup/solve."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_pf_current_source_branching_completion_pair_v5_4 import (
    command, environment, sha256,
)


def tree_inventory(root: Path) -> dict:
    records = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        records.append({
            "relative_path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    fingerprint = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"root": str(root.resolve()), "files": records, "fingerprint": fingerprint}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-worktree", type=Path, required=True)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--mechanical-config", type=Path, required=True)
    parser.add_argument("--migrated-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    execution = args.execution_worktree.resolve()
    migrated_root = args.migrated_root.resolve()
    checkpoints = {
        "control_max1": migrated_root / "control_max1/step0000001_migrated_v5_4.json",
        "enabled_max2": migrated_root / "enabled_max2/step0000001_migrated_v5_4.json",
    }
    for checkpoint in checkpoints.values():
        if not checkpoint.is_file():
            raise RuntimeError(f"missing migrated checkpoint: {checkpoint}")

    package_before = tree_inventory(migrated_root)
    results = {}
    with tempfile.TemporaryDirectory(prefix="pf_v5_4_1_production_restore_sentinel_") as temporary:
        temp_root = Path(temporary)
        for role, maximum_fronts in (("control_max1", 1), ("enabled_max2", 2)):
            case_root = temp_root / role
            cache_root = case_root / "fresh_destination_cache"
            sentinel_path = temp_root / f"{role}_sentinel.json"
            case_env = environment(execution, args.family.resolve(), args.mechanical_config.resolve(), cache_root)
            case_env["PF_V5_4_1_RESTORE_ONLY_SENTINEL_OUT"] = str(sentinel_path)
            case_env["PF_V5_4_1_SENTINEL_PROCESS"] = "1"
            case_command = command(
                output=case_root / "production_entrypoint_output",
                checkpoint=checkpoints[role], family=args.family.resolve(),
                maximum_fronts=maximum_fronts,
            )
            completed = subprocess.run(
                case_command, cwd=execution, env=case_env,
                capture_output=True, text=True, check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"restore-only production entrypoint failed for {role}:\n"
                    f"stdout={completed.stdout}\nstderr={completed.stderr}"
                )
            if not sentinel_path.is_file():
                raise RuntimeError(f"production entrypoint did not publish {role} sentinel")
            record = json.loads(sentinel_path.read_text())
            if record.get("qualification") != "PASS":
                raise RuntimeError(f"restore-only sentinel did not pass for {role}")
            cache_files = [str(path.relative_to(cache_root)) for path in cache_root.rglob("*") if path.is_file()]
            if cache_files:
                raise RuntimeError(f"provider lookup populated restore-only cache: {cache_files}")
            results[role] = {
                **record,
                "maximum_fronts": maximum_fronts,
                "actual_entrypoint_module": "arrhenius_fracture.sharp_front_current_source_branching_audited",
                "command": case_command,
                "environment_fingerprint": hashlib.sha256(
                    json.dumps(case_env, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "entrypoint_returncode": completed.returncode,
                "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
                "stderr_sha256": hashlib.sha256(completed.stderr.encode()).hexdigest(),
                "destination_cache_files_after_restore": cache_files,
            }
    package_after = tree_inventory(migrated_root)
    package_immutable = package_before == package_after
    if not package_immutable:
        raise RuntimeError("migrated checkpoint package changed during restore sentinel")
    payload = {
        "schema": "pf_branching_production_restore_only_sentinel_v5_4_1/2",
        "qualification": "PASS",
        "boundary": "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS",
        "execution_worktree": str(execution),
        "execution_head": subprocess.check_output(
            ("git", "-C", str(execution), "rev-parse", "HEAD"), text=True
        ).strip(),
        "execution_tree": subprocess.check_output(
            ("git", "-C", str(execution), "rev-parse", "HEAD^{tree}"), text=True
        ).strip(),
        "migrated_package_before": package_before,
        "migrated_package_after": package_after,
        "migrated_package_immutable": package_immutable,
        "roles": results,
        "actual_production_entrypoint_reached": True,
        "entrypoint_restore_processes": 2,
        "pf_workers_started": 0,
        "provider_lookups_performed": 0,
        "mechanics_solves_performed": 0,
        "accepted_intervals_begun": 0,
        "stochastic_updates_performed": 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
