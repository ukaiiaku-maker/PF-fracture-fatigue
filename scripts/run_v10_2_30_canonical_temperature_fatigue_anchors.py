#!/usr/bin/env python3
"""Freeze and run the bounded canonical temperature-fatigue anchor matrix."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/mechanical_transfer_temperature_anchors_v1"
MATRIX = OUT / "canonical_temperature_fatigue_anchor_preflight.csv"
REGISTRY = ROOT / "arrhenius_fracture/data/materials/v10_2_27_paper_four_class_registry.csv"
FAMILY = Path(
    "/Volumes/Data/Data/Nanopillar_calculation/"
    "PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/"
    "v10_2_28_kernel_cache/"
    "4fa015d77f1aadf05f77f550366f64cd611f537ae716bbd47870bf9e6fe2f873/"
    "family.json"
)
PYTHON = Path(
    "/opt/homebrew/Caskroom/miniconda/base/envs/"
    "arrhenius-sharp-front-v10-codex/bin/python"
)
BRANCH = "codex/v10.2.30-joint-fracture-fatigue-archetype-atlas"
JOBS = OUT / "canonical_temperature_fatigue_job_registry.csv"
FREEZE = OUT / "canonical_temperature_fatigue_launch_freeze.json"
STATE = OUT / "canonical_temperature_fatigue_controller_state.json"
PHYSICAL = OUT / "physical_canonical_temperature_fatigue"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, path)


def _tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _registry_options() -> dict[str, str]:
    with REGISTRY.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {row["candidate_id"]: row["option_key"] for row in rows}
    if len(result) != 4:
        raise SystemExit("canonical registry is not the exact four-row production registry")
    return result


def build_jobs(head: str) -> list[dict]:
    matrix = pd.read_csv(MATRIX)
    selected = matrix[matrix.preflight_launch_eligible].copy()
    if len(matrix) != 36 or len(selected) != 25:
        raise SystemExit("canonical preflight population drift")
    options = _registry_options()
    jobs: list[dict] = []
    for row in selected.sort_values(
        ["registry_role", "temperature_K", "Kmax_MPa_sqrt_m"]
    ).to_dict("records"):
        candidate = str(row["candidate_id"])
        option = options.get(candidate)
        if option is None:
            raise SystemExit(f"candidate absent from canonical production registry: {candidate}")
        temperature = int(round(float(row["temperature_K"])))
        Kmax = float(row["Kmax_MPa_sqrt_m"])
        R = float(row["R"])
        deltaK = (1.0 - R) * Kmax
        job_id = (
            f"{row['registry_role']}__T{temperature:04d}K__"
            f"Kmax{_tag(Kmax)}__R{_tag(R)}__seed1720"
        )
        result = PHYSICAL / str(row["registry_role"]) / f"T_{temperature:04d}K" / f"Kmax_{_tag(Kmax)}"
        jobs.append({
            "job_id": job_id,
            "registry_role": row["registry_role"],
            "candidate_id": candidate,
            "option_key": option,
            "row_sha256": row["row_sha256"],
            "temperature_K": temperature,
            "temperature_role": row["temperature_role"],
            "Kmax_MPa_sqrt_m": Kmax,
            "DeltaK_MPa_sqrt_m": deltaK,
            "R": R,
            "frequency_Hz": float(row["frequency_Hz"]),
            "n_bins": int(row["n_bins"]),
            "seed": 1720,
            "target_extension_um": float(row["target_extension_um"]),
            "maximum_cycles": float(row["maximum_cycles"]),
            "fresh_virgin_required": True,
            "resume_permitted": False,
            "production_acceleration": "QUALIFIED_EVENT_TO_EVENT_HIGH_CYCLE_ENGINE",
            "solver_head": head,
            "result_path": str(result.resolve()),
            "status": "PENDING",
            "exit_code": "",
            "wall_seconds": "",
        })
    if len({job["job_id"] for job in jobs}) != len(jobs):
        raise SystemExit("duplicate canonical temperature job ID")
    return jobs


def repository_preflight(head: str) -> None:
    if git("branch", "--show-current") != BRANCH:
        raise SystemExit("wrong branch for canonical temperature anchors")
    if git("rev-parse", "HEAD") != head:
        raise SystemExit("HEAD differs from the frozen launch HEAD")
    if git("status", "--porcelain"):
        raise SystemExit("physical launch requires a clean worktree")
    for path in (MATRIX, REGISTRY, FAMILY, PYTHON):
        if not path.is_file():
            raise SystemExit(f"missing launch dependency: {path}")


def freeze(head: str) -> None:
    repository_preflight(head)
    if JOBS.exists() or FREEZE.exists() or PHYSICAL.exists():
        raise SystemExit("canonical temperature launch was already frozen or started")
    jobs = build_jobs(head)
    atomic_csv(JOBS, jobs)
    payload = {
        "schema": "v10.2.30_canonical_temperature_fatigue_launch_freeze_v1",
        "created_utc": now(),
        "branch": BRANCH,
        "head": head,
        "job_count": len(jobs),
        "temperature_count_per_candidate": 3,
        "seed": 1720,
        "R": 0.1,
        "frequency_Hz": 1000.0,
        "n_bins": 80,
        "target_extension_um": 100.0,
        "maximum_cycles": 1.0e12,
        "fresh_virgin_required": True,
        "resume_permitted": False,
        "matrix": str(MATRIX.resolve()),
        "matrix_sha256": sha(MATRIX),
        "registry": str(REGISTRY.resolve()),
        "registry_sha256": sha(REGISTRY),
        "kernel_family": str(FAMILY.resolve()),
        "kernel_family_sha256": sha(FAMILY),
        "job_registry_sha256_at_freeze": sha(JOBS),
        "new_PF_FEM_or_CZM_calculations": False,
        "physical_launch_utc": None,
    }
    atomic_json(FREEZE, payload)
    atomic_json(STATE, {
        "schema": "v10.2.30_canonical_temperature_fatigue_controller_v1",
        "phase": "FROZEN_PENDING",
        "expected_head": head,
        "active_worker_count": 0,
        "status_counts": {"PENDING": len(jobs)},
        "last_update_utc": now(),
    })
    print(json.dumps({"status": "PASS", "frozen_jobs": len(jobs), "head": head}))


def launch_preflight(head: str) -> dict:
    repository_preflight(head)
    if not FREEZE.is_file() or not JOBS.is_file():
        raise SystemExit("run freeze before physical launch")
    payload = json.loads(FREEZE.read_text())
    checks = (
        payload["head"] == head,
        payload["matrix_sha256"] == sha(MATRIX),
        payload["registry_sha256"] == sha(REGISTRY),
        payload["kernel_family_sha256"] == sha(FAMILY),
        payload["job_count"] == 25,
        payload["fresh_virgin_required"] is True,
        payload["resume_permitted"] is False,
    )
    if not all(checks):
        raise SystemExit("canonical temperature launch freeze mismatch")
    return payload


def classify(path: Path, returncode: int) -> str:
    summary_path = path / "developed_fatigue_growth_summary.json"
    checkpoint = path / "high_cycle_live_checkpoint.json"
    if returncode == 0 and summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        if bool(summary.get("target_reached")):
            return "COMPLETE_TARGET_REACHED"
        control_path = path / "v10_2_30_fixed_deltaK_control.json"
        control = json.loads(control_path.read_text()) if control_path.is_file() else {}
        censor = str(control.get("fatigue_censor_status", control.get("censor_status", "")))
        if "censor" in censor.lower() or float(summary.get("cycles_consumed", 0.0)) >= 1.0e12:
            return "COMPLETE_PHYSICAL_CYCLE_CENSOR"
        if int(summary.get("event_count", 0)) > 0:
            return "COMPLETE_PARTIAL_PHYSICAL_GROWTH"
        return "INVALID_NONTERMINAL_ZERO_EVENT"
    if checkpoint.is_file():
        return "NUMERICAL_NONTERMINATION_WITH_DIAGNOSTIC_CHECKPOINT"
    if returncode == 2:
        return "LAUNCH_CONTRACT_FAILURE"
    return "NUMERICAL_FAILURE"


def run_one(job: dict, head: str, maximum_wall_seconds: int) -> dict:
    job = dict(job)
    path = Path(str(job["result_path"]))
    if path.exists():
        job["status"] = "INVALID_DUPLICATE_OUTPUT_PATH"
        return job
    environment = os.environ.copy()
    environment.update({
        "PYTHON_BIN": str(PYTHON),
        "CONDA_ENV": "arrhenius-sharp-front-v10-codex",
        "CONDA_DEFAULT_ENV": "arrhenius-sharp-front-v10-codex",
        "EXPECTED_BRANCH": BRANCH,
        "EXPECTED_HEAD": head,
        "FAMILY_JSON": str(FAMILY),
        "PARAMETER_OPTION": str(job["option_key"]),
        "TARGET_DELTAK": f"{float(job['DeltaK_MPa_sqrt_m']):.17g}",
        "R_RATIO": f"{float(job['R']):.17g}",
        "TEMPERATURE_K": str(int(job["temperature_K"])),
        "FREQUENCY_HZ": f"{float(job['frequency_Hz']):.17g}",
        "TARGET_EXT_UM": f"{float(job['target_extension_um']):.17g}",
        "CYCLES_MAX": f"{float(job['maximum_cycles']):.17g}",
        "HAZARD_SEED": str(int(job["seed"])),
        "MAX_WALL_SECONDS": str(maximum_wall_seconds),
        "TARGET_FRACTION": "canonical_temperature_anchor",
        "RUN_LABEL": str(job["job_id"]),
        "OUTROOT": str(path),
    })
    environment.pop("V10230_RESTART_CHECKPOINT_DIR", None)
    started = time.monotonic()
    completed = subprocess.run(
        ["bash", "scripts/run_v10_2_30_weakt_high_cycle_1e12.sh"],
        cwd=ROOT, env=environment, check=False,
    )
    job["wall_seconds"] = time.monotonic() - started
    job["exit_code"] = completed.returncode
    job["status"] = classify(path, completed.returncode)
    return job


def run(head: str, workers: int, maximum_wall_seconds: int) -> None:
    payload = launch_preflight(head)
    rows = pd.read_csv(JOBS, keep_default_na=False).to_dict("records")
    pending = [row for row in rows if row["status"] == "PENDING"]
    if not pending:
        print(json.dumps({"status": "NO_PENDING_JOBS"}))
        return
    if payload["physical_launch_utc"] is None:
        payload["physical_launch_utc"] = now()
        atomic_json(FREEZE, payload)
    state = json.loads(STATE.read_text())
    state.update({
        "phase": "RUNNING",
        "active_worker_count": min(workers, len(pending)),
        "last_update_utc": now(),
    })
    atomic_json(STATE, state)
    by_id = {str(row["job_id"]): row for row in rows}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_one, row, head, maximum_wall_seconds): row
            for row in pending
        }
        for future in as_completed(futures):
            result = future.result()
            by_id[str(result["job_id"])] = result
            atomic_csv(JOBS, list(by_id.values()))
            print(json.dumps({
                "job_id": result["job_id"],
                "status": result["status"],
                "wall_seconds": result["wall_seconds"],
            }), flush=True)
    rows = list(by_id.values())
    counts = pd.Series([row["status"] for row in rows]).value_counts().to_dict()
    state.update({
        "phase": "TERMINAL",
        "active_worker_count": 0,
        "status_counts": counts,
        "last_update_utc": now(),
    })
    atomic_json(STATE, state)


def status() -> None:
    if STATE.is_file():
        print(STATE.read_text(), end="")
    else:
        print(json.dumps({"phase": "NOT_FROZEN"}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "status"))
    parser.add_argument("--expected-head")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--maximum-wall-seconds", type=int, default=43200)
    args = parser.parse_args()
    if args.command == "status":
        status()
        return 0
    if not args.expected_head:
        raise SystemExit("--expected-head is required")
    if args.workers < 1 or args.workers > 3:
        raise SystemExit("--workers must be between 1 and 3")
    if args.maximum_wall_seconds < 60:
        raise SystemExit("--maximum-wall-seconds must be at least 60")
    if args.command == "freeze":
        freeze(args.expected_head)
    else:
        run(args.expected_head, args.workers, args.maximum_wall_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
