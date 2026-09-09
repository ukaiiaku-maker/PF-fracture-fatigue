#!/usr/bin/env python3
"""Prepare, but deliberately do not execute, future generic N=1/N=2 parity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ENTRY = "arrhenius_fracture.sharp_front_current_source_multifront_v12"
BOUNDARY = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"


def commands(python: Path, control_checkpoint: Path, enabled_checkpoint: Path):
    common = [str(python), "-m", ENTRY, "--source-only-preflight"]
    return {
        "control_n1": common + [
            "--branching-mode", "disabled", "--front-resource-limit", "1",
            "--branch-transaction-limit", "none",
            "--compatibility-checkpoint", str(control_checkpoint),
        ],
        "enabled_n2": common + [
            "--branching-mode", "mechanistic", "--front-resource-limit", "2",
            "--branch-transaction-limit", "none",
            "--compatibility-checkpoint", str(enabled_checkpoint),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--enabled-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if args.launch:
        raise RuntimeError("V6_SOURCE_ONLY_MISSION_FORBIDS_PF_FEM_LAUNCH")
    payload = {
        "schema": "pf_general_multifront_v5_4_1_parity_launcher_v6/1",
        "qualification": "PREPARED_NOT_EXECUTED",
        "boundary": BOUNDARY,
        "commands": commands(args.python, args.control_checkpoint, args.enabled_checkpoint),
        "heavy_pf_run_executed": False,
        "generic_driver_physics_path_specialized_by_front_count": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
