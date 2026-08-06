#!/usr/bin/env python3
"""Run one bounded v11 campaign case with atomic status bookkeeping."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--orientation", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=("mechanistic", "control"), default="mechanistic")
    parser.add_argument("--target-um", type=float, default=25.0)
    parser.add_argument("--steps", type=int, default=2000000)
    args = parser.parse_args(argv)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    module = (
        "arrhenius_fracture.sharp_front_v11_branching_audited"
        if args.mode == "mechanistic" else "arrhenius_fracture.sharp_front_v10_2_28_audited"
    )
    command = [sys.executable, "-m", module]
    if args.mode == "mechanistic":
        command += ["--mechanistic-branching", "--maximum-fronts", "2"]
    else:
        command += ["--max-fronts", "1"]
    command += [
        "--mode", "2d",
        "--parameter-registry", "arrhenius_fracture/data/materials/v10_2_27_v913_four_class_paper_registry.csv",
        "--parameter-option", "v913_paper_weakT01_0129902_persistent_sites",
        "--temperatures", "700", "--steps", str(args.steps), "--nx", "36", "--ny", "72",
        "--dU", "2e-7", "--dt", "8.4", "--n-stagger", "1", "--tip-h-fine", "1e-6",
        "--tip-ratio", "1.20", "--da-phys", "5e-6", "--target-crack-extension-um", str(args.target_um),
        "--front-state-model", "moving_pz", "--tip-source-model", "continuum",
        "--tip-kinetics-mode", "moving_velocity", "--bulk-plasticity-mode", "tip_only",
        "--directional-j-mode", "root_signed", "--tip-plasticity", "--active-shielding",
        "--signed-active-shielding", "--mobile-shield-fraction", "0", "--no-wake-shielding",
        "--crystal-aniso", "--crystal-compete", "--crystal-theta-deg", str(args.orientation),
        "--crystal-material", "w", "--j-decomposition", "cluster", "--crack-backend", "sharp_wake",
        "--adaptive-events", "--adaptive-event-target", "0.15",
        "--out", str(out),
    ]
    environment = os.environ.copy()
    environment["CLEAVAGE_HAZARD_SEED"] = str(args.seed)
    environment["PARAMETER_CAMPAIGN"] = "1"
    status = {
        "schema": "v11.branching-case-status/1", "status": "active", "pid": os.getpid(),
        "start_time": datetime.now(timezone.utc).isoformat(), "command": command,
        "orientation_deg": args.orientation, "hazard_seed": args.seed, "mode": args.mode,
    }
    atomic_json(out / "case_status.json", status)
    result = subprocess.run(command, env=environment)
    if result.returncode == 0 and args.mode == "control":
        atomic_json(out / "run_complete.json", {
            "schema": "v11.branching-run-complete/1", "status": "target_reached",
            "final_checkpoint": None,
            "validation": {"branching_disabled": True, "provider_transition": False},
        })
    completion = {}
    try:
        completion = json.loads((out / "run_complete.json").read_text())
    except (OSError, json.JSONDecodeError):
        pass
    validation = completion.get("validation", {})
    status.update({
        "status": "completed" if result.returncode == 0 else "failed",
        "returncode": result.returncode, "end_time": datetime.now(timezone.utc).isoformat(),
        "provider_transition": (out / "provider_transitions.csv").exists(),
        "latest_event": "branch" if (out / "branch_events.csv").exists() else None,
        "live_fem_solve_count": int(validation.get("live_fem_solve_count", 0)),
    })
    atomic_json(out / "case_status.json", status)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
