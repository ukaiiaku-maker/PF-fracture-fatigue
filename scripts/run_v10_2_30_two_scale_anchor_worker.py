#!/usr/bin/env python3
"""Execute one fresh, exact-only two-scale local anchor contract."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    output = Path(contract["result_path"])
    if output.exists():
        raise SystemExit("fresh anchor worker refuses an existing result path")
    if contract.get("resume") or not contract.get("fresh_virgin_start"):
        raise SystemExit("invalid anchor contract: fresh non-resumed execution required")
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(contract["environment"])
    for key in tuple(environment):
        if "RESUME" in key.upper() or "RESTART" in key.upper():
            environment.pop(key, None)
    started = time.time()
    with Path(contract["log_path"]).open("w") as log:
        process = subprocess.run(
            ["bash", "scripts/run_v10_2_30_weakt_high_cycle_1e12.sh"],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    atomic_json(
        Path(contract["terminal_path"]),
        {
            "schema": "two_scale_anchor_worker_terminal_v1",
            "job_id": contract["job_id"],
            "pid": os.getpid(),
            "exit_code": process.returncode,
            "wall_seconds": time.time() - started,
            "fresh_virgin_start": True,
            "resume": False,
        },
    )
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
