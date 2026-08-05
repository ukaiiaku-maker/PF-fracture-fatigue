#!/usr/bin/env python3
"""Read-only, failure-tolerant campaign monitoring helpers for v10.2.30."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import pandas as pd


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {} if default is None else default


def pid_alive(pid) -> bool | None:
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def validate_campaign_path(repository: Path, campaign: Path) -> Path:
    repo, selected = repository.resolve(), campaign.resolve()
    runs = repo / "runs"
    if selected == runs or runs not in selected.parents:
        raise ValueError("CAMPAIGN_ROOT must be a child of REPOSITORY_ROOT/runs")
    return selected


def combined_checkpoint(case_output: Path) -> dict:
    descriptor = read_json(case_output / "run_state_checkpoint.json")
    generation = descriptor.get("generation")
    result = {"generation": generation, "valid": False, "outer": {}, "kinetic": {}}
    if descriptor.get("schema") != "v10.2.30_combined_outer_kinetic_run_state_v2" or not generation:
        return result
    folder = case_output / "run_state_generations" / generation
    try:
        for name, expected in descriptor.get("files", {}).items():
            path = folder / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                return result
        result["outer"] = read_json(folder / "outer.json")
        result["kinetic"] = read_json(folder / "kinetic.json")
        result["valid"] = bool(result["outer"] and result["kinetic"] and (folder / "state.npz").is_file())
    except OSError:
        pass
    return result


def classify(status: str, summary: dict, control: dict, events: int) -> str:
    if summary.get("target_reached") is True:
        return "completed_growth"
    censor = control.get("censor_status", "")
    if "right_censored" in censor:
        return "right_censored_after_growth" if events else "right_censored_no_growth"
    aliases = {"completed": "completed_growth", "censored": "right_censored_no_growth",
               "restartable": "incomplete_restartable", "blocked-before-launch": "blocked"}
    return aliases.get(status, status or "unknown")


def case_row(case: Path, matrix_row: dict | None = None, now: float | None = None,
             stale_seconds: float = 900.0) -> tuple[dict, list[str]]:
    now = time.time() if now is None else now
    row = matrix_row or {}; output = case / "output" if (case / "output").exists() else case
    status = read_json(case / "qualification_status.json")
    live = read_json(output / "high_cycle_live_checkpoint.json")
    liveness = read_json(output / "qualification_liveness.json")
    summary = read_json(output / "developed_fatigue_growth_summary.json")
    control = read_json(output / "v10_2_30_fixed_deltaK_control.json")
    geometry = read_json(output / "stochastic_avalanche_geometry_events.json", [])
    checkpoint = combined_checkpoint(output)
    outer, kinetic = checkpoint["outer"], checkpoint["kinetic"]
    stochastic = live.get("stochastic") or kinetic.get("stochastic", {})
    event_count = len(geometry) if isinstance(geometry, list) else int(summary.get("event_count", 0) or 0)
    cycles = live.get("cycles_from_engine_time", summary.get("cycles_consumed", outer.get("cycles_total")))
    extension = summary.get("final_projected_extension_um")
    if extension is None:
        extension = outer.get("geometry", {}).get("projected_extension_m")
        extension = None if extension is None else float(extension) * 1e6
    pid = status.get("pid"); alive = pid_alive(pid)
    physical = liveness.get("latest_physical_progress_timestamp", status.get("latest_physical_progress_timestamp"))
    live_stamp = liveness.get("latest_liveness_timestamp", live.get("timestamp_unix_s", status.get("latest_liveness_timestamp")))
    action, threshold = stochastic.get("hazard_action_current"), stochastic.get("hazard_threshold_action")
    warnings = []
    state = classify(status.get("status", ""), summary, control, event_count)
    if state == "running" and alive is False: warnings.append("running case has dead worker PID")
    if live_stamp and now - float(live_stamp) > stale_seconds: warnings.append("checkpoint/liveness is stale")
    if physical and live_stamp and now-float(physical)>stale_seconds and now-float(live_stamp)<=stale_seconds:
        warnings.append("physical progress stale while liveness is current")
    if (case / "output/run_state_checkpoint.json").exists() and not checkpoint["valid"]:
        warnings.append("combined checkpoint manifest is invalid")
    if state == "incomplete_restartable" and not checkpoint["valid"]:
        warnings.append("restartable case lacks a valid combined checkpoint")
    horizon = row.get("cycle_horizon", control.get("cycles_max"))
    if state == "running" and horizon and cycles and float(cycles) > float(horizon):
        warnings.append("active case exceeds configured cycle horizon")
    size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) if output.exists() else 0
    provenance = summary.get("provenance", {})
    option = row.get("parameter_option") or provenance.get("parameter_option")
    result = {
        "case": case.name, "class": row.get("label"), "option": option,
        "fraction": row.get("fraction"), "deltaK": row.get("deltaK_MPa_sqrt_m"),
        "seed": row.get("seed"), "status": state, "pid": pid, "pid_alive": alive,
        "cycles": cycles, "cycle_horizon": horizon, "event_count": event_count,
        "extension_um": extension, "target_um": row.get("target_extension_um", 100.0),
        "phase": liveness.get("phase", live.get("reason", status.get("current_phase"))),
        "hazard_action": action, "threshold": threshold,
        "action_fraction": float(action)/float(threshold) if action is not None and threshold else None,
        "restart_count": status.get("restart_count", 0), "physical_progress_timestamp": physical,
        "liveness_timestamp": live_stamp, "checkpoint_generation": checkpoint["generation"],
        "checkpoint_valid": checkpoint["valid"], "output_size_bytes": size,
        "developed_da_dN": summary.get("developed_interval", {}).get("da_dN"),
        "git_head": provenance.get("git_head") or control.get("git_head"),
    }
    return result, warnings


def campaign_snapshot(repository: Path, campaign: Path, stale_seconds: float = 900.0) -> dict:
    root = validate_campaign_path(repository, campaign)
    matrix = read_json(root / "dense_deltaK_matrix.json") or read_json(root / "qualification_matrix.json")
    rows = matrix.get("cases", []); lookup = {r.get("case"): r for r in rows}
    case_dirs = sorted({p.parent for p in root.glob("*/qualification_status.json")} | {root/r["case"] for r in rows if r.get("case")})
    records, warnings = [], []
    for case in case_dirs:
        record, issues = case_row(case, lookup.get(case.name), stale_seconds=stale_seconds)
        records.append(record); warnings.extend(f"{case.name}: {issue}" for issue in issues)
    order = {"running": 0, "pending": 1, "incomplete_restartable": 2}
    frame = pd.DataFrame(records)
    if not frame.empty:
        frame["_order"] = frame.status.map(lambda x: order.get(x, 3))
        frame = frame.sort_values(["_order", "class", "fraction"], ascending=[True, True, False]).drop(columns="_order")
    lock = read_json(root / "qualification_supervisor.lock.json")
    launcher = read_json(root / "launcher.json")
    owner = lock or launcher
    free = shutil.disk_usage("/Volumes/Data").free
    minimum = matrix.get("minimum_free_gib", 10.0)
    if free / 1024**3 < minimum: warnings.append("free space is below supervisor threshold")
    heads = {r.get("git_head") for r in records if r.get("git_head")}
    if len(heads) > 1: warnings.append("mixed Git provenance across cases")
    statuses = frame.status.value_counts().to_dict() if not frame.empty else {}
    return {"root": root, "cases": frame, "warnings": warnings, "owner": owner,
            "supervisor_alive": pid_alive(owner.get("pid")), "counts": statuses,
            "active_workers": int(statuses.get("running", 0)), "free_bytes": free,
            "disk_bytes": sum(r.get("output_size_bytes", 0) for r in records), "matrix": matrix}


def tail(path: Path, lines: int = 80) -> str:
    try:
        with path.open(errors="replace") as stream:
            return "".join(stream.readlines()[-lines:])
    except OSError:
        return ""


def recent_event_details(case: Path, limit: int = 20) -> list[dict]:
    """Return a bounded authoritative event tail without mutating artifacts."""
    output = case / "output" if (case / "output").exists() else case
    summary = read_json(output / "developed_fatigue_growth_summary.json")
    events = summary.get("event_measurements", [])
    if isinstance(events, list) and events:
        return events[-limit:]
    geometry = read_json(output / "stochastic_avalanche_geometry_events.json", [])
    return geometry[-limit:] if isinstance(geometry, list) else []
