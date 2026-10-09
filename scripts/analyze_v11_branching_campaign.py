#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
import sys


def _rows(path):
    if not path.exists(): return []
    with path.open(newline="") as stream: return list(csv.DictReader(stream))


def analyze(root):
    root = Path(root).resolve(); launcher = json.loads((root / "launcher.json").read_text())
    cases = launcher.get("planned_cases", []); table = []; events = []
    for spec in cases:
        spec = {"case_id": spec} if isinstance(spec, str) else dict(spec); directory = root / spec["case_id"]
        complete = json.loads((directory / "run_complete.json").read_text()) if (directory / "run_complete.json").exists() else {}
        local = _rows(directory / "branch_events.csv")
        for row in local: events.append({"case_id": spec["case_id"], **row})
        table.append({**spec, "completion_status": complete.get("status", "incomplete"), "branched": bool(local), "branch_event_count": len(local)})
    orientations = {}
    for row in table:
        key = str(row.get("orientation_deg", "unknown")); orientations.setdefault(key, []).append(row)
    orientation_summary = [{
        "orientation_deg": key, "case_count": len(values), "branch_count": sum(v["branched"] for v in values),
        "branch_probability": sum(v["branched"] for v in values) / len(values),
    } for key, values in sorted(orientations.items(), key=lambda item: item[0])]
    summary = {"schema": "v11.branching-campaign-summary/1", "case_count": len(table), "branch_count": sum(r["branched"] for r in table), "orientation_summary": orientation_summary}
    return summary, table, events, orientation_summary


def _write(path, rows):
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1: raise SystemExit("usage: analyze_v11_branching_campaign.py CAMPAIGN_ROOT")
    root = Path(args[0]).resolve(); summary, table, events, orientations = analyze(root)
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write(root / "case_table.csv", table); _write(root / "branch_events_all.csv", events); _write(root / "orientation_summary.csv", orientations)
    _write(root / "energy_margin_summary.csv", []); _write(root / "provider_transition_summary.csv", [])
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__": main()
