#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def _json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def campaign_status(root):
    root = Path(root).resolve()
    launcher = _json(root / "launcher.json") or {}
    cases = launcher.get("planned_cases", [])
    rows = []
    for item in cases:
        case_id = item if isinstance(item, str) else item["case_id"]
        directory = root / case_id
        complete = _json(directory / "run_complete.json")
        status = _json(directory / "case_status.json") or {}
        valid_complete = bool(complete and complete.get("schema") == "v11.branching-run-complete/1")
        pid = status.get("pid")
        active = False
        if pid:
            try:
                os.kill(int(pid), 0); active = True
            except (OSError, ValueError):
                pass
        rows.append({
            "case_id": case_id, "completed": valid_complete, "completion_status": complete.get("status") if valid_complete else None,
            "active": active, "failed": status.get("status") == "failed" or (valid_complete and complete.get("status") == "numerical_failure"),
            "latest_event": status.get("latest_event"), "provider_transition": status.get("provider_transition", False),
            "live_fem_solve_count": int(status.get("live_fem_solve_count", 0)), "pid": pid,
            "branching": (directory / "branch_events.csv").exists() and (directory / "branch_events.csv").stat().st_size > 1,
        })
    usage = subprocess.check_output(("du", "-sk", str(root)), text=True).split()[0] if root.exists() else "0"
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(), "campaign_root": str(root),
        "git_head": launcher.get("git_head"), "planned_cases": len(cases),
        "completed_cases": sum(row["completed"] for row in rows), "active_cases": sum(row["active"] for row in rows),
        "failed_cases": sum(row["failed"] for row in rows),
        "handoff_complete_cases": sum(row["completion_status"] == "branch_cluster_independent_tip_handoff_required" for row in rows),
        "branching_cases": sum(row["branching"] for row in rows),
        "nonbranching_cases": sum(row["completed"] and not row["branching"] for row in rows),
        "provider_transitions": sum(row["provider_transition"] for row in rows),
        "live_fem_solve_count": sum(row["live_fem_solve_count"] for row in rows),
        "disk_usage_kib": int(usage), "active_pids": [row["pid"] for row in rows if row["active"]], "cases": rows,
    }


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise SystemExit("usage: status_v11_branching_campaign.py CAMPAIGN_ROOT")
    print(json.dumps(campaign_status(args[0]), indent=2, sort_keys=True))


if __name__ == "__main__": main()
