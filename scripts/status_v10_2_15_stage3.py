#!/usr/bin/env python3
"""Report launcher, case, and completion status for the Stage 3 campaign."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
from typing import Any

DEFAULT_OUTROOT = Path(
    "runs/v10_2_15_stage3_four_option_monotonic_500um_theta45_1x_v1"
)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _pid_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _process_table() -> str:
    completed = subprocess.run(
        ["ps", "-ax", "-o", "pid=,command="],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout if completed.returncode == 0 else ""


def _last_nonempty_line(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        lines = [line.strip() for line in path.read_text(errors="replace").splitlines()]
    except OSError:
        return ""
    return next((line for line in reversed(lines) if line), "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outroot", type=Path, default=DEFAULT_OUTROOT)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--tail", type=int, default=12)
    args = parser.parse_args()

    outroot = args.outroot.expanduser().resolve()
    log_path = (
        args.log.expanduser().resolve()
        if args.log is not None
        else Path(str(outroot) + ".nohup.log")
    )
    status_path = outroot / "overnight_status.json"
    status = _load_json(status_path) or {}

    pid_candidates: list[int] = []
    for path in (
        outroot / "overnight_launcher.pid",
        Path(str(outroot) + ".pid"),
    ):
        if path.is_file():
            try:
                pid_candidates.append(int(path.read_text().strip()))
            except (OSError, ValueError):
                pass
    if isinstance(status.get("launcher_pid"), int):
        pid_candidates.append(int(status["launcher_pid"]))
    launcher_pid = next((pid for pid in pid_candidates if _pid_alive(pid)), None)

    plan_path = outroot / "stage3_campaign_plan.tsv"
    planned: list[dict[str, str]] = []
    if plan_path.is_file():
        with plan_path.open(newline="") as handle:
            planned = list(csv.DictReader(handle, delimiter="\t"))

    process_table = _process_table()
    counts = {
        "complete": 0,
        "incomplete": 0,
        "censored": 0,
        "failed": 0,
        "active": 0,
        "queued": 0,
    }
    active_rows: list[tuple[str, str]] = []
    for row in planned:
        case_root = Path(row["case_root"]).expanduser().resolve()
        case_status = _load_json(case_root / "stage3_case_status.json")
        if case_status:
            category = {
                "complete_target_extension": "complete",
                "incomplete_after_first_passage": "incomplete",
                "right_censored_no_first_passage": "censored",
            }.get(str(case_status.get("status")))
            if category:
                counts[category] += 1
                continue
        if (case_root / "RUN_FAILED").is_file():
            counts["failed"] += 1
            continue
        if str(case_root) in process_table:
            counts["active"] += 1
            active_rows.append(
                (
                    f"{row['option_key']} T={row['temperature_K']}K",
                    _last_nonempty_line(case_root / "run.log"),
                )
            )
            continue
        counts["queued"] += 1

    state = str(status.get("state", "unknown"))
    if launcher_pid is None and state in {"starting", "assembling", "running"}:
        state = "stopped_or_failed"
    if not status and launcher_pid is None:
        state = "not_running"

    print("Stage 3 campaign status")
    print(f"  state:       {state}")
    print(f"  launcher:    {'alive PID ' + str(launcher_pid) if launcher_pid else 'not running'}")
    print(f"  outroot:     {outroot}")
    if status.get("message"):
        print(f"  message:     {status['message']}")
    if status.get("updated_utc"):
        print(f"  updated:     {status['updated_utc']}")
    print(f"  planned:     {len(planned)}")
    print(
        "  cases:       "
        f"active={counts['active']} complete={counts['complete']} "
        f"incomplete={counts['incomplete']} censored={counts['censored']} "
        f"failed={counts['failed']} queued={counts['queued']}"
    )

    if active_rows:
        print("  active cases:")
        for label, line in active_rows:
            suffix = f" | {line}" if line else ""
            print(f"    - {label}{suffix}")

    if log_path.is_file() and args.tail > 0:
        lines = log_path.read_text(errors="replace").splitlines()[-args.tail :]
        print(f"  latest log ({log_path}):")
        for line in lines:
            print(f"    {line}")

    return 0 if launcher_pid is not None or state == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
