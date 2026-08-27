#!/usr/bin/env python3
"""Persistent, fresh-only controller for mandatory two-scale local anchors."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from arrhenius_fracture.two_scale_virtual_ct_v10230 import LogPchipRateSurface

PYTHON = "/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python"
BRANCH = "codex/v10.2.30-two-scale-virtual-CT"
SOURCE = Path("runs/A_native_PT03_PT08_R_nominal_deltaK_v1")
DEFAULT_ROOT = Path("runs/A_native_two_scale_virtual_CT_v1")
FAMILY = "/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/v10_2_28_kernel_cache/4fa015d77f1aadf05f77f550366f64cd611f537ae716bbd47870bf9e6fe2f873/family.json"
QUALIFIED_SOLVER_HEAD = "94871be15702e7fb85116b92af62c1226c61be42"
SOLVER_SHA256 = "c15a957161e8cdb43bf944243d8a18a64839dbfe0518aa2d1c5f0c93773b817b"
SOURCE_FILES = (
    "A_PT03_PT08_R_developed_points.csv",
    "A_PT03_PT08_R_second_seed_points.csv",
    "A_PT03_PT08_constant_load_CT_windows.csv",
    "A_PT03_PT08_local_to_nominal_K_transfer.csv",
    "A_PT03_PT08_R_event_results.csv",
    "A_PT03_PT08_R_final_decision.json",
    "deltaK_semantics_audit.json",
)
RS = (-0.95, 0.1, 0.5)
ANCHOR_K = (13.5, 21.0, 24.3)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, path)


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], text=True).strip()


def alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, PermissionError, TypeError, ValueError):
        return False


def source_hashes() -> dict[str, str]:
    hashes = {}
    for name in SOURCE_FILES:
        path = SOURCE / name
        if not path.is_file():
            raise RuntimeError(f"missing frozen source artifact: {path}")
        hashes[name] = sha256(path)
    return hashes


def surface_rows(frame: pd.DataFrame, option: str) -> list[dict]:
    selected = frame[(frame.option == option) & (frame.stage.str.contains("PRIMARY"))].copy()
    selected = selected[selected.seed == 1720]
    selected = selected[selected.Kmax_MPa_sqrt_m.isin((12.0, 15.0, 18.0, 24.0))]
    if len(selected) != 12:
        raise RuntimeError(f"expected twelve stationary source rows for {option}, got {len(selected)}")
    if not (selected.target_reached.all() and selected.stable_growth.all() and (~selected.resumed).all()):
        raise RuntimeError(f"nonphysical/nonstationary row in {option} surface source")
    return selected.sort_values(["R", "Kmax_MPa_sqrt_m"]).to_dict("records")


def initialize(root: Path, head: str) -> list[dict]:
    root.mkdir(parents=True, exist_ok=False)
    for folder in ("anchors", "job_contracts", "worker_terminals", "controller_logs"):
        (root / folder).mkdir()
    hashes = source_hashes()
    frame = pd.read_csv(SOURCE / "A_PT03_PT08_R_developed_points.csv")
    all_rows = frame.sort_values(["option", "R", "Kmax_MPa_sqrt_m"]).copy()
    all_rows["two_scale_role"] = all_rows.option.map(
        lambda option: "PRIMARY_SURFACE" if option == "A_NATIVE" else "DIAGNOSTIC_OVERLAY"
    )
    all_rows.to_csv(root / "two_scale_source_rows.csv", index=False)
    atomic_json(root / "two_scale_source_hashes.json", hashes)
    atomic_json(
        root / "two_scale_source_manifest.json",
        {
            "schema": "two_scale_source_manifest_v1",
            "created_unix_ns": time.time_ns(),
            "source_root": str(SOURCE.resolve()),
            "source_branch": "codex/v10.2.30-R-ratio-nominal-deltaK",
            "source_analysis_head": "4b554006d60539d7dec659f4bfe4bf298728780a",
            "qualified_solver_head": QUALIFIED_SOLVER_HEAD,
            "production_solver_sha256": SOLVER_SHA256,
            "source_hashes": hashes,
            "source_row_count": len(all_rows),
            "held_out_constant_load_rows_used_for_fit": 0,
            "tip_radius_used_in_nominal_K": False,
        },
    )
    native_rows = surface_rows(frame, "A_NATIVE")
    surface = LogPchipRateSurface(native_rows, option="A_NATIVE", version="v0")
    frozen_ns = time.time_ns()
    v0 = {
        "schema": "log_log_PCHIP_local_rate_surface_v1",
        "option": "A_NATIVE",
        "version": "v0",
        "frozen_unix_ns": frozen_ns,
        "interpolation": "PCHIP_IN_LN_K_AND_LN_RATE_PER_MEASURED_R;LINEAR_IN_LN_RATE_BETWEEN_R",
        "global_Paris_law": False,
        "extrapolation": False,
        "K_domain_MPa_sqrt_m": [surface.K_min, surface.K_max],
        "R_domain": [surface.R_min, surface.R_max],
        "source_rows": surface.nodes(),
        "held_out_constant_load_fit_rows": 0,
    }
    atomic_json(root / "A_NATIVE_rate_surface_v0.json", v0)
    pd.DataFrame(surface.nodes()).to_parquet(root / "A_NATIVE_rate_surface_v0.parquet", index=False)
    loo = []
    for R in RS:
        Rrows = [row for row in native_rows if math.isclose(float(row["R"]), R)]
        for held_K in (15.0, 18.0):
            train = [row for row in Rrows if not math.isclose(float(row["Kmax_MPa_sqrt_m"]), held_K)]
            held = next(row for row in Rrows if math.isclose(float(row["Kmax_MPa_sqrt_m"]), held_K))
            trial = LogPchipRateSurface(train, option="A_NATIVE", version="v0_LOO")
            evaluation = trial.evaluate(held_K, R)
            predicted = float(evaluation.rate_m_per_cycle)
            actual = float(held["developed_da_dN"])
            neighbors = sorted(float(row["developed_da_dN"]) for row in train)
            loo.append(
                {
                    "R": R,
                    "held_Kmax_MPa_sqrt_m": held_K,
                    "predicted_da_dN": predicted,
                    "actual_da_dN": actual,
                    "epsilon_log_decade": math.log10(predicted / actual),
                    "abs_epsilon_log_decade": abs(math.log10(predicted / actual)),
                    "relative_rate_error": predicted / actual - 1,
                    "predicted_local_slope": float(evaluation.local_slope),
                    "monotonic": True,
                    "within_neighbor_rate_bounds": neighbors[0] <= predicted <= neighbors[-1],
                }
            )
    atomic_csv(root / "A_NATIVE_rate_surface_v0_validation.csv", loo)
    predictions = []
    for R in RS:
        for K in ANCHOR_K:
            row = surface.prospective_anchor_prediction(K, R)
            row.update(
                {
                    "option": "A_NATIVE",
                    "surface_version": "v0",
                    "prediction_frozen_unix_ns": frozen_ns,
                    "result_read_before_prediction": False,
                    "result_path_existed_when_predicted": False,
                }
            )
            predictions.append(row)
    atomic_csv(root / "A_NATIVE_anchor_predictions_v0.csv", predictions)
    prediction_sha = sha256(root / "A_NATIVE_anchor_predictions_v0.csv")
    jobs = []
    for R in RS:
        for K in ANCHOR_K:
            job_id = f"anchor__A_NATIVE__R{R:g}__Kmax{K:g}__seed1720"
            jobs.append(
                {
                    "job_id": job_id,
                    "option": "A_NATIVE",
                    "R": R,
                    "Kmax_MPa_sqrt_m": K,
                    "deltaK_driver_MPa_sqrt_m": (1 - R) * K,
                    "seed": 1720,
                    "n_bins": 80,
                    "temperature_K": 300,
                    "frequency_Hz": 1000,
                    "target_extension_um": 100,
                    "cycles_max": 1_000_000,
                    "result_path": str((root / "anchors" / job_id).resolve()),
                    "status": "PENDING",
                    "attempt": 0,
                    "pid": None,
                    "resumed": False,
                    "reused": False,
                    "acceleration_mode": "explicit_only",
                    "prediction_file_sha256": prediction_sha,
                    "prediction_frozen_unix_ns": frozen_ns,
                    "contract_path": "",
                    "terminal_path": "",
                    "log_path": "",
                }
            )
    atomic_csv(root / "two_scale_anchor_job_registry.csv", jobs)
    atomic_json(
        root / "two_scale_controller_state.json",
        {"schema": "two_scale_controller_state_v1", "phase": "PREDICTIONS_FROZEN", "launch_head": head},
    )
    return jobs


def verify_frozen_source(root: Path) -> None:
    expected = json.loads((root / "two_scale_source_hashes.json").read_text())
    if source_hashes() != expected:
        raise RuntimeError("frozen two-scale source artifact changed")
    surface = json.loads((root / "A_NATIVE_rate_surface_v0.json").read_text())
    if surface.get("version") != "v0" or surface.get("extrapolation") is not False:
        raise RuntimeError("frozen v0 surface contract invalid")


def make_contract(job: dict, root: Path, head: str) -> Path:
    if Path(job["result_path"]).exists():
        raise RuntimeError(f"fresh anchor result path already exists: {job['job_id']}")
    attempt = int(job["attempt"]) + 1
    log = (root / "controller_logs" / f"{job['job_id']}__attempt{attempt}.log").resolve()
    terminal = (root / "worker_terminals" / f"{job['job_id']}__attempt{attempt}.json").resolve()
    contract_path = root / "job_contracts" / f"{job['job_id']}__attempt{attempt}.json"
    environment = {
        "PYTHON_BIN": PYTHON,
        "CONDA_ENV": "arrhenius-sharp-front-v10-codex",
        "CONDA_DEFAULT_ENV": "arrhenius-sharp-front-v10-codex",
        "EXPECTED_BRANCH": BRANCH,
        "EXPECTED_HEAD": head,
        "FAMILY_JSON": FAMILY,
        "V10230_ENTRY_MODULE": "arrhenius_fracture.sharp_front_v10_2_30_candidate_fixed_deltaK",
        "V10230_CANDIDATE_REGISTRY": str((SOURCE / "A_PT03_PT08_R_registry.csv").resolve()),
        "V10230_CANDIDATE_SELECTION": str((SOURCE / "A_PT03_PT08_R_selection.json").resolve()),
        "PARAMETER_OPTION": "A_NATIVE",
        "TARGET_DELTAK": f"{float(job['deltaK_driver_MPa_sqrt_m']):.17g}",
        "R_RATIO": f"{float(job['R']):.17g}",
        "TARGET_FRACTION": "TWO_SCALE_ANCHOR",
        "RUN_LABEL": job["job_id"],
        "TARGET_EXT_UM": "100",
        "CYCLES_MAX": str(int(job["cycles_max"])),
        "HAZARD_SEED": "1720",
        "MAX_WALL_SECONDS": "43200",
        "OUTROOT": job["result_path"],
        "V10230_HIGH_CYCLE_EXPLICIT_ONLY": "1",
    }
    payload = {
        "schema": "two_scale_anchor_contract_v1",
        "job_id": job["job_id"],
        "attempt": attempt,
        "created_unix_ns": time.time_ns(),
        "result_path": job["result_path"],
        "log_path": str(log),
        "terminal_path": str(terminal),
        "launch_head": head,
        "qualified_solver_head": QUALIFIED_SOLVER_HEAD,
        "production_solver_sha256": SOLVER_SHA256,
        "prediction_file_sha256": job["prediction_file_sha256"],
        "prediction_frozen_unix_ns": int(job["prediction_frozen_unix_ns"]),
        "fresh_virgin_start": True,
        "resume": False,
        "environment": environment,
    }
    atomic_json(contract_path, payload)
    job.update(
        {
            "attempt": attempt,
            "contract_path": str(contract_path.resolve()),
            "terminal_path": str(terminal),
            "log_path": str(log),
        }
    )
    return contract_path


def valid_terminal(job: dict) -> bool:
    summary_path = Path(job["result_path"]) / "developed_fatigue_growth_summary.json"
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text())
    return (
        bool(summary.get("target_reached"))
        and bool(summary.get("stable_growth_provisional"))
        and int(summary.get("event_count", 0)) >= 10
        and int(summary.get("restart_count", 0)) == 0
    )


def run_anchors(root: Path, jobs: list[dict], head: str, workers: int) -> None:
    registry = root / "two_scale_anchor_job_registry.csv"
    while True:
        for job in [row for row in jobs if row["status"] == "RUNNING"]:
            terminal = Path(job["terminal_path"])
            if terminal.is_file() or not alive(job["pid"]):
                data = json.loads(terminal.read_text()) if terminal.is_file() else {"exit_code": None}
                job["exit_code"] = data.get("exit_code")
                job["wall_seconds"] = data.get("wall_seconds")
                job["pid"] = None
                job["status"] = "PHYSICAL_TARGET_REACHED" if valid_terminal(job) else "INVALID_OR_NONTERMINAL"
                atomic_csv(registry, jobs)
                if job["status"] != "PHYSICAL_TARGET_REACHED":
                    raise RuntimeError(f"fresh exact anchor failed closed: {job['job_id']}")
        active = [row for row in jobs if row["status"] == "RUNNING"]
        pending = [row for row in jobs if row["status"] == "PENDING"]
        while pending and len(active) < workers:
            job = pending.pop(0)
            contract_path = make_contract(job, root, head)
            process = subprocess.Popen(
                [PYTHON, "scripts/run_v10_2_30_two_scale_anchor_worker.py", "--contract", str(contract_path)],
                start_new_session=True,
            )
            job["pid"] = process.pid
            job["status"] = "RUNNING"
            active.append(job)
            atomic_csv(registry, jobs)
        if not pending and not active:
            return
        time.sleep(5)


def controller_lock(root: Path) -> Path:
    path = root / "controller.lock"
    if path.exists():
        old = json.loads(path.read_text())
        if alive(old.get("pid")):
            raise SystemExit(f"controller already active pid={old['pid']}")
        path.unlink()
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, json.dumps({"pid": os.getpid()}).encode())
    os.close(descriptor)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--initialize-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        raise SystemExit("workers must be 1..3")
    branch, head = git("branch", "--show-current"), git("rev-parse", "HEAD")
    if branch != BRANCH or git("status", "--short"):
        raise SystemExit("two-scale controller requires the requested clean branch")
    root = args.root.resolve()
    jobs = (
        pd.read_csv(root / "two_scale_anchor_job_registry.csv").to_dict("records")
        if (root / "two_scale_anchor_job_registry.csv").is_file()
        else initialize(root, head)
    )
    verify_frozen_source(root)
    if args.initialize_only:
        return 0
    interrupted = [row["job_id"] for row in jobs if row["status"] == "RUNNING" and not alive(row["pid"])]
    if interrupted:
        raise RuntimeError(f"interrupted physical anchor cannot resume or rerun in place: {interrupted}")
    lock = controller_lock(root)
    try:
        atomic_json(root / "two_scale_controller_state.json", {"phase": "MANDATORY_ANCHORS", "pid": os.getpid(), "head": head})
        run_anchors(root, jobs, head, args.workers)
        atomic_json(root / "two_scale_controller_state.json", {"phase": "ANCHORS_COMPLETE", "pid": os.getpid(), "head": head})
    finally:
        if lock.exists():
            lock.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
