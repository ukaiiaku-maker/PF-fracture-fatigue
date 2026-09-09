#!/usr/bin/env python3
"""V5.4.1 completion-pair preflight; execution remains unauthorized."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_pf_current_source_branching_completion_pair_v5_4 import (
    FAMILY_PHYSICS, FAMILY_SHA, MECHANICAL_SHA, SOURCE_FAMILY_SHA,
    checkpoint_record, command, environment, heavy_process_inventory, require,
    stable_json_hash,
)


EXECUTION_COMMIT = "ae9a06d8c42287428e917baef143b0c1142cefd8"
EXECUTION_TREE = "2d0edbc7c7b51d81ab5678065ca68a1cef45d134"
BOUNDARY = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(root), *args), text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-worktree", type=Path, required=True)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument("--source-family", type=Path, required=True)
    parser.add_argument("--mechanical-config", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--enabled-checkpoint", type=Path, required=True)
    parser.add_argument("--production-restore-sentinel", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-out", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.launch:
        raise SystemExit(
            "V5.4.1 is a completion seal only: completion execution is not authorized"
        )

    execution = args.execution_worktree.resolve()
    if git(execution, "rev-parse", "HEAD") != EXECUTION_COMMIT:
        raise RuntimeError("execution worktree is not the V5.4.1 reviewed source commit")
    if git(execution, "rev-parse", "HEAD^{tree}") != EXECUTION_TREE:
        raise RuntimeError("execution worktree tree is not the V5.4.1 reviewed source tree")
    if git(execution, "status", "--porcelain"):
        raise RuntimeError("V5.4.1 execution worktree must be clean")
    require(args.family, FAMILY_SHA, "V5.3 append-only family")
    require(args.source_family, SOURCE_FAMILY_SHA, "V5.2 source family")
    require(args.mechanical_config, MECHANICAL_SHA, "mechanical configuration")
    checkpoints = {
        "control_max1": checkpoint_record(args.control_checkpoint, "control_max1", 1),
        "enabled_max2": checkpoint_record(args.enabled_checkpoint, "enabled_max2", 2),
    }
    sentinel = json.loads(args.production_restore_sentinel.read_text())
    if sentinel.get("qualification") != "PASS":
        raise RuntimeError("production restore-only sentinel is not qualified")
    if sentinel.get("execution_head") != EXECUTION_COMMIT:
        raise RuntimeError("production restore sentinel used a different source commit")
    if sentinel.get("pf_workers_started") != 0 or sentinel.get("mechanics_solves_performed") != 0:
        raise RuntimeError("production restore sentinel crossed its no-solve boundary")
    for role in checkpoints:
        record = sentinel.get("roles", {}).get(role, {})
        if record.get("qualification") != "PASS" or not all(record.get("checks", {}).values()):
            raise RuntimeError(f"production restore sentinel failed for {role}")

    cases = []
    for role, maximum in (("control_max1", 1), ("enabled_max2", 2)):
        case = args.output_root / f"theta40_v5_4_1_{role}_seed3621"
        cache = case / "live_kernel_cache"
        cases.append({
            "role": role,
            "maximum_fronts": maximum,
            "candidate_id": "oneD_v2_focused_weak_T_0016",
            "temperature_K": 700,
            "theta_deg": 40,
            "hazard_seed": 3621,
            "target_projected_extension_um": 300,
            "output_directory": str(case.resolve()),
            "output_exists": case.exists(),
            "command": command(
                output=case, checkpoint=Path(checkpoints[role]["manifest"]),
                family=args.family.resolve(), maximum_fronts=maximum,
            ),
            "environment": environment(
                execution, args.family.resolve(), args.mechanical_config.resolve(), cache
            ),
        })
    payload = {
        "schema": "pf_branching_completion_preflight_v5_4_1/1",
        "qualification": "PASS",
        "boundary": BOUNDARY,
        "mode": "DRY_RUN_ONLY",
        "execution_commit": EXECUTION_COMMIT,
        "execution_tree": EXECUTION_TREE,
        "review_record_parent": "f9d6d115110bb0f915d6403f1a83671698a1ab3d",
        "qualified_v5_4_execution_parent": "fb6516bb6a9f770fee899d382a902cf0bdf1e701",
        "family_sha256": FAMILY_SHA,
        "family_physics_fingerprint": FAMILY_PHYSICS,
        "source_family_sha256": SOURCE_FAMILY_SHA,
        "mechanical_configuration_sha256": MECHANICAL_SHA,
        "checkpoints": checkpoints,
        "production_restore_sentinel": {
            "path": str(args.production_restore_sentinel.resolve()),
            "sha256": hashlib.sha256(args.production_restore_sentinel.read_bytes()).hexdigest(),
            "qualification": sentinel["qualification"],
        },
        "cases": cases,
        "environment_fingerprints": {
            case["role"]: stable_json_hash(case["environment"]) for case in cases
        },
        "heavy_process_inventory": heavy_process_inventory(),
        "maximum_fronts_supported": [1, 2],
        "maximum_fronts_greater_than_two_supported": False,
        "completion_authorized": False,
        "completion_executed": False,
        "thousand_um_extension_authorized": False,
        "predictive_branching_physics_validated": False,
        "pf_workers_started": 0,
        "mechanics_solves_performed": 0,
    }
    args.preflight_out.parent.mkdir(parents=True, exist_ok=True)
    args.preflight_out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(args.preflight_out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
