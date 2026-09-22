#!/usr/bin/env python3
"""Branch the natural reference at the last valid pre-site-2-birth checkpoint."""
from __future__ import annotations

import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE_JOB = ROOT / "runs/managed_natural_void_v1_20260919"
TARGET_JOB = ROOT / "runs/managed_natural_void_multisite_v1_20260920"
CHECKPOINT = "restart_002600ms.json"


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    source = SOURCE_JOB / "output"
    target = TARGET_JOB / "output"
    if target.exists():
        raise RuntimeError(f"corrected branch already exists: {target}")
    target.mkdir(parents=True)

    manifest = json.loads((source / CHECKPOINT).read_text())
    state_file = str(manifest["state_file"])
    for name in ("declared_inputs.json", CHECKPOINT, state_file):
        shutil.copy2(source / name, target / name)

    old = json.loads((source / "progress.json").read_text())
    history = [
        row for row in old["history"]
        if float(row["end_time_s"]) <= 2.59999 + 1.0e-12
    ]
    if not history or float(history[-1]["end_time_s"]) != 2.59999:
        raise RuntimeError("the required 2.59999 s accepted prefix is unavailable")
    progress = {
        "accepted_time_s": 2.59999,
        "history": history,
        "last_accepted_event": history[-1]["selected_events"][-1]
            if history[-1]["selected_events"] else None,
        "latest_restart": CHECKPOINT,
        "latest_restart_state_file": state_file,
        "next_checkpoint_time_s": 3.0,
        "phase": history[-1]["state"],
        "solver_activity": {
            "mesh_elements": history[-1]["mesh_elements"],
            "mesh_nodes": history[-1]["mesh_nodes"],
        },
        "stop_reason": "CORRECTED_BRANCH_PREPARED_FROM_VALID_PREFIX",
        "provenance": {
            "source_job": str(SOURCE_JOB),
            "source_terminal_result_preserved": str(source / "result.json"),
            "branch_reason": "MULTISITE_SUBGRID_LIFECYCLE_DISPATCH_REPAIR",
            "last_valid_prefix_s": 2.59999,
        },
    }
    dump(target / "progress.json", progress)
    dump(TARGET_JOB / "branch.json", progress["provenance"])
    print(target)
    print("HISTORY_ROWS", len(history))
    print("ACCEPTED_TIME_S", progress["accepted_time_s"])


if __name__ == "__main__":
    main()
