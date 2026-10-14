#!/usr/bin/env python3
"""Compare native runs that differ only in retained-output policy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "native-output-policy-independence/1"
PHYSICS_ARTIFACTS = (
    "summary.json",
    "steps_0700K.csv",
    "fronts_0700K.csv",
    "branch_diagnostics_0700K.csv",
    "crack_path_700K.csv",
    "crack_path_front0_700K.csv",
    "kinetic_tip_cell_audit_v101.json",
    "v10_0_1_driver_modes.json",
    "v10_1_driver_modes.json",
)
POLICY_FIELDS = ("save_snapshots", "snapshot_cols", "no_mpz_state_output", "no_plots")
NONPHYSICAL_ARGUMENT_FIELDS = set(POLICY_FIELDS) | {"out", "print_every"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_runs(sparse_root: Path, full_root: Path) -> dict[str, Any]:
    sparse_args = json.loads((sparse_root / "run_args.json").read_text())
    full_args = json.loads((full_root / "run_args.json").read_text())
    sparse_physics = {k: v for k, v in sparse_args.items() if k not in NONPHYSICAL_ARGUMENT_FIELDS}
    full_physics = {k: v for k, v in full_args.items() if k not in NONPHYSICAL_ARGUMENT_FIELDS}
    if sparse_physics != full_physics:
        changed = sorted(
            key for key in set(sparse_physics) | set(full_physics)
            if sparse_physics.get(key) != full_physics.get(key)
        )
        raise ValueError(f"physical/non-output arguments differ: {changed}")

    policy = {
        "sparse": {key: sparse_args.get(key) for key in POLICY_FIELDS},
        "full": {key: full_args.get(key) for key in POLICY_FIELDS},
    }
    if policy["sparse"] == policy["full"]:
        raise ValueError("output policies are identical")

    artifacts: dict[str, Any] = {}
    mismatches: list[str] = []
    for name in PHYSICS_ARTIFACTS:
        sparse_path = sparse_root / name
        full_path = full_root / name
        if not sparse_path.is_file() or not full_path.is_file():
            raise FileNotFoundError(f"required calculated-state artifact is missing: {name}")
        sparse_hash = sha256(sparse_path)
        full_hash = sha256(full_path)
        identical = sparse_hash == full_hash
        if not identical:
            mismatches.append(name)
        artifacts[name] = {
            "sparse_sha256": sparse_hash,
            "full_sha256": full_hash,
            "byte_identical": identical,
        }

    full_only_rendered = sorted(
        path.name for path in full_root.glob("*.png")
        if not (sparse_root / path.name).is_file()
    )
    if not full_only_rendered:
        raise ValueError("full-output run did not create additional rendered fields/diagnostics")
    return {
        "schema": SCHEMA,
        "decision": "PASS" if not mismatches else "FAIL_STATE_DEPENDS_ON_OUTPUT_POLICY",
        "qualification_scope": (
            "short native 2-D ceramic event trajectory; two accepted 5 um events, "
            "not a production material prediction"
        ),
        "physics_arguments_identical": True,
        "output_policies": policy,
        "calculated_state_artifacts": artifacts,
        "mismatches": mismatches,
        "full_only_rendered_outputs": full_only_rendered,
        "interpretation": (
            "selected field/diagnostic retention changed stored outputs but did not "
            "change the byte-level calculated state and history"
        ),
        "production_execution_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse-root", type=Path, required=True)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare_runs(args.sparse_root, args.full_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
