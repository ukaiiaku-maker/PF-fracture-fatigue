#!/usr/bin/env python3
"""Restart-safe supervisor for the fixed eight-case v11 production matrix."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = "arrhenius_fracture/data/materials/v10_2_27_v913_four_class_paper_registry.csv"
CASES = (
    ("peak_0300K", "peak", "v913_paper_peak01_0242980_persistent_sites", 300),
    ("peak_1000K", "peak", "v913_paper_peak01_0242980_persistent_sites", 1000),
    ("dbtt_0300K", "dbtt", "v913_paper_dbtt01_0202500_persistent_sites", 300),
    ("dbtt_1000K", "dbtt", "v913_paper_dbtt01_0202500_persistent_sites", 1000),
    ("weakt_0300K", "weakt", "v913_paper_weakT01_0129902_persistent_sites", 300),
    ("weakt_1000K", "weakt", "v913_paper_weakT01_0129902_persistent_sites", 1000),
    ("ceramic_0300K", "ceramic", "v913_paper_ceramic01_0077080_persistent_sites", 300),
    ("ceramic_1000K", "ceramic", "v913_paper_ceramic01_0077080_persistent_sites", 1000),
)


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def valid_checkpoint(case_root: Path) -> Path | None:
    path = case_root / "checkpoint/latest.json"
    if not path.exists():
        return None
    manifest = load(path)
    state = path.with_name(str(manifest.get("state_file", "")))
    return path if state.is_file() else None


def case_row(spec, campaign_root: Path):
    case_id, material, option, temperature = spec
    root = campaign_root / case_id
    completion = load(root / "run_complete.json")
    status = load(root / "case_status.json")
    checkpoint = valid_checkpoint(root)
    complete = completion.get("status") == "target_reached"
    pid = status.get("pid")
    active = False
    if status.get("status") == "running" and pid:
        try:
            os.kill(int(pid), 0)
            active = True
        except (OSError, ValueError):
            pass
    if complete:
        classification = "complete"
    elif active:
        classification = "running"
    elif status.get("status") == "failed":
        classification = "failed"
    elif checkpoint:
        classification = "restartable"
    else:
        classification = "not_started"
    manifest = load(checkpoint) if checkpoint else {}
    counters = manifest.get("event_counters", {})
    return {
        "case_id": case_id, "parameter_class": material, "parameter_option": option,
        "temperature_K": temperature, "seed": 3621, "status": classification,
        "latest_extension_um": float(manifest.get("physical_extension_m", 0.0)) * 1e6,
        "target_extension_um": 1000.0,
        "branch_birth_count": int(manifest.get("committed_branch_birth_count", 0)),
        "active_tip_count": int(manifest.get("active_front_count", 0)),
        "coalescence_count": int(counters.get("coalescence_count", 0)),
        "accepted_steps": int(counters.get("accepted_steps", 0)),
        "mesh_generation": int(manifest.get("mesh_generation", 0)),
        "restart_count": int(status.get("restart_count", 0)),
        "last_checkpoint": str(checkpoint or ""), "terminal_reason": status.get("terminal_reason"),
        "pid": pid if active else None,
    }


def write_summary(root: Path):
    rows = [case_row(spec, root) for spec in CASES]
    atomic_json(root / "campaign_status.json", {
        "schema": "v11.four-class-branching-campaign-status/1",
        "updated_at": datetime.now(timezone.utc).isoformat(), "cases": rows,
    })
    return rows


def launch_command(spec, root: Path, family: Path):
    case_id, _, option, temperature = spec
    case_root = root / case_id
    checkpoint = valid_checkpoint(case_root)
    command = [sys.executable, "-m", "arrhenius_fracture.sharp_front_v11_branching_audited",
        "--mechanistic-branching", "--maximum-fronts", "16", "--mode", "2d",
        "--parameter-registry", REGISTRY, "--parameter-option", option,
        "--temperatures", str(temperature), "--steps", "2000000", "--nx", "36", "--ny", "72",
        "--dU", "2e-7", "--dt", "8.4", "--n-stagger", "1", "--tip-h-fine", "1e-6",
        "--tip-ratio", "1.20", "--da-phys", "5e-6", "--target-crack-extension-um", "1000",
        "--front-state-model", "moving_pz", "--tip-source-model", "continuum",
        "--tip-kinetics-mode", "moving_velocity", "--bulk-plasticity-mode", "tip_only",
        "--directional-j-mode", "root_signed", "--tip-plasticity", "--active-shielding",
        "--signed-active-shielding", "--mobile-shield-fraction", "0", "--no-wake-shielding",
        "--crystal-aniso", "--crystal-compete", "--crystal-theta-deg", "30",
        "--crystal-material", "w", "--j-decomposition", "cluster", "--crack-backend", "sharp_wake",
        "--adaptive-events", "--adaptive-event-target", "0.15", "--out", str(case_root)]
    if checkpoint:
        command += ["--v11-restart-checkpoint", str(checkpoint)]
    env = os.environ.copy()
    env.update(CLEAVAGE_HAZARD_SEED="3621", PARAMETER_CAMPAIGN="1",
               SIGNED_KERNEL_FAMILY_JSON=str(family), KERNEL_CACHE_ROOT=str(family.parents[1]))
    prior = load(case_root / "case_status.json")
    case_root.mkdir(parents=True, exist_ok=True)
    log = (case_root / "production.log").open("a")
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    atomic_json(case_root / "case_status.json", {
        "schema": "v11.four-class-case-status/1", "status": "running", "pid": process.pid,
        "restart_count": int(prior.get("restart_count", 0)) + int(checkpoint is not None),
        "launch_time": datetime.now(timezone.utc).isoformat(), "command": command,
        "parameter_option": option, "temperature_K": temperature,
    })
    return process, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--family", type=Path, required=True)
    ap.add_argument("--max-jobs", type=int, default=2)
    ap.add_argument("--initialize-only", action="store_true")
    args = ap.parse_args(); root = args.root.resolve(); family = args.family.resolve()
    if args.max_jobs < 1: raise SystemExit("max-jobs must be positive")
    data = json.loads(family.read_text())
    coverage = max(float(x) for x in data["cumulative_crack_path_extension_levels_m"])
    if coverage < 1.0e-3 or data.get("production_physics_modified") is not False:
        raise SystemExit("kernel family does not satisfy production coverage/physics contract")
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "campaign_manifest.json").exists():
        head = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
        branch = subprocess.check_output(("git", "branch", "--show-current"), cwd=ROOT, text=True).strip()
        atomic_json(root / "campaign_manifest.json", {
            "schema": "v11.four-class-branching-campaign/1", "git_head": head, "branch": branch,
            "hostname": socket.gethostname(), "python_executable": sys.executable,
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"), "parameter_registry": REGISTRY,
            "theta_deg": 30, "seed": 3621, "target_extension_um": 1000,
            "physical_mpz_and_handoff_m": 50e-6, "da_phys_m": 5e-6,
            "provider_identity": "v11_exact_crack_network_live_fem_v1",
            "kernel_family": str(family), "kernel_coverage_m": coverage,
            "launch_timestamp": datetime.now(timezone.utc).isoformat(),
            "maximum_parallel_jobs": args.max_jobs,
            "cases": [dict(case_id=a, parameter_class=b, parameter_option=c, temperature_K=d) for a,b,c,d in CASES],
        })
    write_summary(root)
    if args.initialize_only: return
    running = {}
    while True:
        rows = write_summary(root)
        pending = [spec for spec, row in zip(CASES, rows) if row["status"] in ("not_started", "restartable")]
        while pending and len(running) < args.max_jobs:
            spec = pending.pop(0); process, log = launch_command(spec, root, family)
            running[spec[0]] = (process, log)
        externally_running = any(row["status"] == "running" for row in rows)
        if not running and not pending and not externally_running: break
        time.sleep(10)
        for case_id, (process, log) in list(running.items()):
            code = process.poll()
            if code is None: continue
            log.close(); status_path = root / case_id / "case_status.json"; status = load(status_path)
            completion = load(root / case_id / "run_complete.json")
            status.update(status="complete" if code == 0 and completion.get("status") == "target_reached" else "failed",
                          returncode=code, terminal_reason=completion.get("status") or f"returncode_{code}",
                          end_time=datetime.now(timezone.utc).isoformat())
            atomic_json(status_path, status); del running[case_id]
    write_summary(root)


if __name__ == "__main__": main()
