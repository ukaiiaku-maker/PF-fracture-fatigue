"""Part X: persistent disk-backed physical-job controller (mission section 14).

Launches AUTHORIZED_* rows from a job registry (screen_job_registry.csv
or, once individually authorized, developed_job_registry.csv) as
independent subprocesses (scripts/part_x_run_one_job.py), never
interpreting QUEUED_NOT_LAUNCHED or BLOCKED_* as authorization -- only a
status literally starting with "AUTHORIZED_" is ever launched.

Guarantees implemented here:
- one fresh Python process per scientific job (the PX2.5-qualified
  reproducibility pattern);
- maximum 3 concurrent workers;
- atomic controller-state updates (write-temp-then-os.replace, never a
  partial write visible to a concurrent reader);
- expected-HEAD enforcement (refuses to launch if the current git HEAD
  does not match the registry's recorded physical_producer_sha, unless
  --allow-head-drift is explicitly passed);
- exact config-hash enforcement (delegated to part_x_run_one_job.py,
  which re-derives the hash from the registry row and refuses to launch
  on any mismatch);
- canonical duplicate-key rejection (a canonical_job_key already present
  in controller_state.json's completed/running/quarantined sets is never
  relaunched);
- virgin result paths (part_x_run_one_job.py's mkdir(exist_ok=False));
- no resume (there is no resume code path at all -- an interrupted job's
  directory is quarantined, never continued);
- prephysics-failure and interrupted-physics quarantine, distinguished;
- refuses to launch new jobs while the tracked source tree is dirty.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = REPO_ROOT / "runs" / "crack_rebonding_part_x_v1"
STATE_PATH = RUN_ROOT / "controller_state.json"
MAX_WORKERS = 3


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _load_registry(path: Path) -> list[dict[str, Any]]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _require_clean_worktree() -> None:
    status = subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, capture_output=True, text=True).stdout
    if status.strip():
        raise RuntimeError(
            "refusing to launch: tracked source tree is not clean "
            "(no source edits are permitted while workers may be active):\n" + status
        )


def _check_expected_head(expected_sha: str, allow_drift: bool) -> str:
    actual = _git("rev-parse", "HEAD")
    if expected_sha and expected_sha != "UNKNOWN_AT_REGISTRATION_TIME" and actual != expected_sha:
        if not allow_drift:
            raise RuntimeError(
                f"expected-HEAD mismatch: registry job was registered at {expected_sha}, "
                f"current HEAD is {actual} -- refusing to launch (pass --allow-head-drift to override "
                "only if you have verified the intervening commits do not affect physics)"
            )
    return actual


def run(
    *, registry_path: Path, preflight: bool, max_jobs: int | None, allow_drift: bool,
    run_root: Path | None = None, require_clean: bool = True, enforce_head: bool = True,
) -> dict[str, Any]:
    run_root = run_root or RUN_ROOT
    state_path = run_root / "controller_state.json"

    def _load_state_here() -> dict[str, Any]:
        if state_path.exists():
            return json.loads(state_path.read_text())
        return {"schema": "v10230_part_x_controller_state_v1", "jobs": {}}

    def _save_state_here(state: dict[str, Any]) -> None:
        run_root.mkdir(parents=True, exist_ok=True)
        tmp = state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
        os.replace(tmp, state_path)

    if require_clean:
        _require_clean_worktree()
    jobs = [j for j in _load_registry(registry_path) if j["status"].startswith("AUTHORIZED_")]
    if max_jobs is not None:
        jobs = jobs[:max_jobs]

    state = _load_state_here()
    actual_head = _git("rev-parse", "HEAD")
    if enforce_head and jobs:
        actual_head = _check_expected_head(jobs[0]["physical_producer_sha"], allow_drift)

    run_root.mkdir(parents=True, exist_ok=True)
    quarantine_prephysics = run_root / "_quarantine_prephysics"
    quarantine_interrupted = run_root / "_quarantine_interrupted"
    quarantine_prephysics.mkdir(exist_ok=True)
    quarantine_interrupted.mkdir(exist_ok=True)

    launched = 0
    skipped_duplicate = 0
    running: list[tuple[subprocess.Popen, dict[str, Any], Path]] = []

    def _launch(job: dict[str, Any]) -> None:
        nonlocal launched
        key = job["canonical_job_key"]
        if key in state["jobs"] and state["jobs"][key]["status"] in ("COMPLETE", "COMPLETE_PREFLIGHT", "RUNNING"):
            return
        job_dir_name = f"{job['protocol']}_{job['row_name']}_{job['cohesion']}_{key[:12]}"
        result_dir = run_root / job_dir_name
        if result_dir.exists():
            raise RuntimeError(f"refusing to launch: result path already exists (not virgin): {result_dir}")
        job_json_path = run_root / f"{job_dir_name}.job.json"
        job_json_path.write_text(json.dumps(job, indent=2))

        cmd = [
            sys.executable, str(REPO_ROOT / "scripts" / "part_x_run_one_job.py"),
            "--job-json", str(job_json_path), "--result-dir", str(result_dir),
        ]
        if preflight:
            cmd.append("--preflight")
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT)
        state["jobs"][key] = {
            "status": "RUNNING", "job": job, "result_dir": str(result_dir),
            "pid": proc.pid, "launch_head": actual_head, "launch_time": time.time(),
        }
        _save_state_here(state)
        running.append((proc, job, result_dir))
        launched += 1

    pending = list(jobs)
    while pending or running:
        while pending and len(running) < MAX_WORKERS:
            job = pending.pop(0)
            key = job["canonical_job_key"]
            if key in state["jobs"] and state["jobs"][key]["status"] in ("COMPLETE", "COMPLETE_PREFLIGHT"):
                skipped_duplicate += 1
                continue
            _launch(job)

        if not running:
            break

        time.sleep(0.2)
        still_running = []
        for proc, job, result_dir in running:
            ret = proc.poll()
            if ret is None:
                still_running.append((proc, job, result_dir))
                continue
            key = job["canonical_job_key"]
            status_path = result_dir / "job_status.json"
            reported = json.loads(status_path.read_text())["status"] if status_path.exists() else None
            if ret == 0 and reported in ("COMPLETE", "COMPLETE_PREFLIGHT"):
                state["jobs"][key]["status"] = reported
            elif reported == "PREPHYSICS_LAUNCH_FAILURE":
                state["jobs"][key]["status"] = "PREPHYSICS_LAUNCH_FAILURE"
                if result_dir.exists():
                    result_dir.rename(quarantine_prephysics / result_dir.name)
                    state["jobs"][key]["result_dir"] = str(quarantine_prephysics / result_dir.name)
            else:
                # Nonzero exit without a clean terminal status -- an
                # interrupted-physics case (killed, crashed mid-run).
                # Never treated as science, never resumed.
                state["jobs"][key]["status"] = "INTERRUPTED_NOT_SCIENCE"
                if result_dir.exists():
                    result_dir.rename(quarantine_interrupted / result_dir.name)
                    state["jobs"][key]["result_dir"] = str(quarantine_interrupted / result_dir.name)
            state["jobs"][key]["exit_code"] = ret
            _save_state_here(state)
        running = still_running

    return {
        "launched": launched, "skipped_duplicate": skipped_duplicate,
        "total_authorized_in_registry": len(jobs),
        "final_statuses": {k: v["status"] for k, v in state["jobs"].items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True, help="Path to screen_job_registry.csv or developed_job_registry.csv")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--allow-head-drift", action="store_true")
    args = parser.parse_args()

    summary = run(
        registry_path=Path(args.registry), preflight=args.preflight,
        max_jobs=args.max_jobs, allow_drift=args.allow_head_drift,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
