#!/usr/bin/env python3
"""Launch one detached, resource-bounded natural-void worker."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "runs/managed_natural_void_v1_20260919"
SOURCE = Path(os.environ.get("NATURAL_VOID_SOURCE", JOB / "source"))
OUTPUT = JOB / "output"
LOG = Path(os.environ.get("NATURAL_VOID_LOG", JOB / "worker.log"))
PYTHON = Path(
    "/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python"
)


def main():
    if not (SOURCE / "snapshot_manifest.json").exists():
        raise RuntimeError("immutable source snapshot is missing")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metadata_path = JOB / "job.json"
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        try:
            os.kill(int(old["pid"]), 0)
        except (OSError, ProcessLookupError):
            pass
        else:
            raise RuntimeError(f"managed worker is already live: PID {old['pid']}")
    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "MPLCONFIGDIR": "/private/tmp/mpl-cache",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MAX_WALL_SECONDS": "1800",
        "MAX_OUTPUT_BYTES": str(8 * 1024**3),
        "MAX_SCHEDULER_TRANSACTIONS_PER_CHECKPOINT": os.environ.get(
            "MAX_SCHEDULER_TRANSACTIONS_PER_CHECKPOINT", "32"),
        "NATURAL_VOID_RUN_OUT": str(OUTPUT),
    })
    command = [str(PYTHON), "-u", "scripts/run_natural_void_hypothesis_v1.py"]
    log_handle = LOG.open("ab", buffering=0)
    process = subprocess.Popen(
        command, cwd=SOURCE, env=env, stdout=log_handle,
        stderr=subprocess.STDOUT, start_new_session=True,
    )
    metadata = {
        "schema": "managed-natural-void-job/1",
        "pid": process.pid,
        "command": command,
        "execution_directory": str(SOURCE),
        "output_directory": str(OUTPUT),
        "log": str(LOG),
        "resource_limits": {
            "heavy_workers": 1,
            "blas_threads": 1,
            "wall_time_per_chunk_s": 1800,
            "maximum_output_bytes": 8 * 1024**3,
        },
        "resume_command": (
            f"cd {SOURCE} && NATURAL_VOID_RUN_OUT={OUTPUT} "
            f"MAX_WALL_SECONDS=1800 {PYTHON} -u scripts/run_natural_void_hypothesis_v1.py"
        ),
        "stop_command": f"kill -TERM {process.pid}",
        "source_snapshot_manifest": str(SOURCE / "snapshot_manifest.json"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(process.pid)


if __name__ == "__main__":
    main()
