#!/usr/bin/env python3
"""Restart-safe supervisor for the bounded v10.2.30 four-class qualification."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid

try:
    from scripts.v10230_qualification_family import validate as validate_family
except ModuleNotFoundError:  # direct execution places scripts/ on sys.path
    from v10230_qualification_family import validate as validate_family


OPTIONS = {
    "peak": ("v913_paper_peak01_0242980_persistent_sites", 1720, 21.289546465050222),
    "dbtt": ("v913_paper_dbtt01_0202500_persistent_sites", 1001723, 21.02530765128298),
    "weakT": ("v913_paper_weakT01_0129902_persistent_sites", 2001726, 12.702935563752424),
    "ceramic": ("v913_paper_ceramic01_0077080_persistent_sites", 3001729, 12.259477791864454),
}
FRACTIONS = (0.55, 0.75, 0.95)
TERMINAL = {"completed", "censored"}
QUALIFIED_SIMULATION_HEAD = "24b63a5bfd86a8ea249d457750b14b8c19488973"
LOCK_NAME = "qualification_supervisor.lock.json"
EXPECTED_MATRIX = {
    (label, fraction): critical * fraction
    for label, (_option, _seed, critical) in OPTIONS.items()
    for fraction in FRACTIONS
}


def artifacts(case: Path) -> Path:
    output = case / "output"
    return output if output.exists() else case


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def matrix() -> list[dict]:
    rows = []
    for label, (option, seed, critical) in OPTIONS.items():
        for fraction in FRACTIONS:
            rows.append({
                "case": f"{label}_f{str(fraction).replace('.', 'p')}_seed{seed}",
                "label": label,
                "parameter_option": option,
                "fraction": fraction,
                "deltaK_MPa_sqrt_m": critical * fraction,
                "seed": seed,
            })
    return rows


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {} if default is None else default


def checkpoint_valid(case: Path, row: dict | None = None) -> bool:
    case = artifacts(case)
    try:
        from arrhenius_fracture.run_state_checkpoint_v10230 import (
            load_combined_checkpoint, validate_cross_layer,
        )
        outer, kinetic, _arrays = load_combined_checkpoint(case)
        validate_cross_layer(outer, kinetic)
        metadata = outer.get("geometry", {}).get("mesh_metadata", {})
        if set(("hbar_m", "hbar_tip_m", "tip_reference_centers_m")) - set(metadata):
            return False
        if row is not None:
            recorded = outer.get("case", {})
            expected = {
                "temperature_K": 300.0, "deltaK_MPa_sqrt_m": row["deltaK_MPa_sqrt_m"],
                "R": 0.1, "frequency_Hz": 1000.0, "seed": row["seed"],
            }
            if any(recorded.get(key) != value for key, value in expected.items()):
                return False
            recorded_option = recorded.get("parameter_option")
            if not recorded_option:
                selection = read_json(case / "v10_2_22_parameter_selection.json")
                recorded_option = selection.get("exact_registry_row", {}).get("option_key")
            if recorded_option != row["parameter_option"]:
                return False
        return True
    except (Exception,):
        return False


def progress(case: Path) -> dict:
    case = artifacts(case)
    checkpoint = read_json(case / "high_cycle_live_checkpoint.json")
    summary = read_json(case / "developed_fatigue_growth_summary.json")
    geometry = read_json(case / "stochastic_avalanche_geometry_events.json", [])
    liveness = read_json(case / "qualification_liveness.json")
    return {
        "cycles_reached": checkpoint.get("cycles_from_engine_time", summary.get("cycles_consumed", 0.0)),
        "event_count": len(geometry) if isinstance(geometry, list) else summary.get("event_count", 0),
        "crack_extension_um": summary.get("final_projected_extension_um", 0.0),
        "current_mode": checkpoint.get("mode") or checkpoint.get("reason"),
        "current_phase": liveness.get("phase", "pending"),
        "latest_physical_progress_timestamp": liveness.get("latest_physical_progress_timestamp", 0.0),
        "latest_liveness_timestamp": liveness.get("latest_liveness_timestamp", 0.0),
        "checkpoint_valid": checkpoint_valid(case),
    }


def status_path(case: Path) -> Path:
    return case / "qualification_status.json"


def set_status(case: Path, state: str, **extra) -> dict:
    old = read_json(status_path(case))
    payload = {**old, **progress(case), **extra, "status": state, "updated_unix_s": time.time()}
    atomic_json(status_path(case), payload)
    return payload


def classify(case: Path, row: dict | None = None) -> str:
    old = read_json(status_path(case))
    if old.get("status") in TERMINAL | {"blocked-before-launch"}:
        return old["status"]
    output = artifacts(case)
    if (output / "exit_code.txt").is_file():
        try:
            code = int((output / "exit_code.txt").read_text().strip())
        except ValueError:
            code = 1
        control = read_json(output / "v10_2_30_fixed_deltaK_control.json")
        summary = read_json(output / "developed_fatigue_growth_summary.json")
        if code == 0 and control.get("censor_status") == "right_censored_no_event":
            return "censored"
        if code == 0 and summary.get("status") == "growth_target_reached" and summary.get("target_reached") is True:
            return "completed"
        return "restartable" if checkpoint_valid(case, row) else "failed"
    return "restartable" if checkpoint_valid(case, row) else "pending"


def stale(case: Path, now: float, interval: float) -> bool:
    stamp = float(progress(case)["latest_liveness_timestamp"] or 0.0)
    return stamp > 0.0 and now - stamp > interval


def process_identity(pid: int) -> str | None:
    try:
        result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="], text=True,
                                capture_output=True)
        value = result.stdout.strip()
        return value or None
    except OSError:
        try:
            os.kill(pid, 0)
            return f"live-pid:{pid}"
        except OSError:
            return None


def acquire_lock(root: Path, head: str, recover: bool = False) -> dict:
    path = root / LOCK_NAME; host = socket.gethostname(); pid = os.getpid()
    payload = {"schema": "v10.2.30_qualification_supervisor_lock_v1", "token": uuid.uuid4().hex,
               "pid": pid, "process_start_identity": process_identity(pid), "hostname": host,
               "campaign_root": str(root.resolve()), "git_head": head, "launch_timestamp_unix_s": time.time()}
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        old = read_json(path)
        live = old.get("hostname") == host and process_identity(int(old.get("pid", -1))) == old.get("process_start_identity")
        if live or not recover:
            raise RuntimeError(f"campaign ownership lock exists ({'live' if live else 'stale; use --recover-stale-lock'}): {old}")
        path.unlink()
        return acquire_lock(root, head, False)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    return payload


def release_lock(root: Path, token: str) -> None:
    path = root / LOCK_NAME
    if read_json(path).get("token") == token:
        path.unlink(missing_ok=True)


def infer_phase(log_text: str) -> str:
    markers = (("RESTART restored", "restart_reconstruction"), ("robust forward VHCF", "exact_event_localization"),
               ("dmd_event_guard", "exact_event_localization"), ("<< ADVANCE", "geometry_commit"),
               ("wrote avalanche geometry", "checkpoint_serialization"), ("developed_fatigue", "final_analysis"))
    for marker, phase in markers:
        if marker in log_text:
            return phase
    return "high_cycle_evolution"


def update_liveness(case: Path, process: subprocess.Popen, tracker: dict) -> None:
    output = artifacts(case); log = output / "run.log"; now = time.time()
    checkpoint = read_json(output / "high_cycle_live_checkpoint.json")
    signature = (checkpoint.get("cycles_from_engine_time"), len(read_json(output / "stochastic_avalanche_geometry_events.json", [])))
    log_stamp = log.stat().st_mtime if log.exists() else 0.0
    if signature != tracker.get("physical_signature"):
        tracker["physical_signature"] = signature
        tracker["physical_timestamp"] = now
        tracker["liveness_timestamp"] = now
    if log_stamp != tracker.get("log_stamp") or process.poll() is not None:
        tracker["log_stamp"] = log_stamp; tracker["liveness_timestamp"] = now
    tail = log.read_text(errors="replace")[-12000:] if log.exists() else ""
    atomic_json(output / "qualification_liveness.json", {
        "schema": "v10.2.30_qualification_liveness_v1", "pid": process.pid,
        "phase": infer_phase(tail), "latest_physical_progress_timestamp": tracker.get("physical_timestamp", now),
        "latest_liveness_timestamp": tracker.get("liveness_timestamp", now),
        "physical_progress": {"cycles": signature[0], "event_count": signature[1]},
    })


def free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / 1024**3


def worker_command(row: dict, case: Path, args) -> list[str]:
    if args.smoke_worker:
        return [sys.executable, __file__, "smoke-worker", str(case)]
    return ["bash", "scripts/run_v10_2_30_weakt_high_cycle_1e12.sh"]


def worker_environment(row: dict, case: Path, args, restarting: bool) -> dict:
    env = os.environ.copy()
    repository = Path(__file__).parents[1]
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=repository, text=True
    ).strip()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    env.update({
        "PYTHON_BIN": sys.executable,
        "CONDA_ENV": "arrhenius-sharp-front-v10-codex",
        "CONDA_DEFAULT_ENV": "arrhenius-sharp-front-v10-codex",
        "EXPECTED_BRANCH": branch,
        "EXPECTED_HEAD": head,
        "MAX_WALL_SECONDS": "31536000",
        "PARAMETER_OPTION": row["parameter_option"],
        "TARGET_DELTAK": f'{row["deltaK_MPa_sqrt_m"]:.17g}',
        "TARGET_FRACTION": str(row["fraction"]),
        "RUN_LABEL": row["case"],
        "HAZARD_SEED": str(row["seed"]),
        "TARGET_EXT_UM": str(args.target_extension_um),
        "CYCLES_MAX": str(args.cycles_max),
        "OUTROOT": str(case if args.smoke_worker else case / "output"),
        "V10230_HIGH_CYCLE_CHECKPOINT_DIR": str(
            case if args.smoke_worker else case / "output"
        ),
        "FAMILY_JSON": str(args.family_path),
    })
    if restarting:
        env["V10230_RESTART_CHECKPOINT_DIR"] = str(
            case if args.smoke_worker else case / "output"
        )
    return env


def run(args) -> int:
    repository = Path(__file__).parents[1]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    if (not args.smoke_worker and
            subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True).strip()):
        raise RuntimeError("qualification launch requires a clean worktree")
    family = validate_family(args.family_descriptor)
    args.family_path = Path(family["path"])
    rows = matrix()
    if any(row["deltaK_MPa_sqrt_m"] != EXPECTED_MATRIX[(row["label"], row["fraction"])] for row in rows):
        raise RuntimeError("qualification matrix differs from committed full-precision values")
    root = args.root.resolve(); root.mkdir(parents=True, exist_ok=True)
    ownership = acquire_lock(root, head, args.recover_stale_lock)
    atomic_json(root / "qualification_matrix.json", {"schema": "v10.2.30_qualification_matrix_v1",
        "launch_git_head": head, "qualified_simulation_head": QUALIFIED_SIMULATION_HEAD,
        "family": family, "cases": rows})
    atomic_json(root / "launcher.json", ownership)
    for row in matrix():
        case = root / row["case"]; case.mkdir(exist_ok=True)
        if not status_path(case).exists():
            set_status(case, "pending", pid=None, restart_count=0)
    stop_requested = False
    def stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    active: dict[str, tuple[subprocess.Popen, Path, object, dict]] = {}
    maximum_active_observed = 0
    pending = list(matrix())
    try:
        while pending or active:
            for name, (process, case, log, tracker) in list(active.items()):
                update_liveness(case, process, tracker)
                state = None
                if process.poll() is not None:
                    row = next(row for row in rows if row["case"] == name)
                    log.close(); state = classify(case, row)
                    if state not in TERMINAL:
                        state = "restartable" if checkpoint_valid(case, row) else "failed"
                    set_status(case, state, pid=process.pid, exit_code=process.returncode)
                    del active[name]
                    if state == "restartable" and int(read_json(status_path(case)).get("restart_count", 0)) < args.max_restarts:
                        pending.append(row)
                elif stale(case, time.time(), args.no_progress_seconds):
                    os.killpg(process.pid, signal.SIGTERM)
                    try: process.wait(timeout=args.term_grace_seconds)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait()
                    log.close()
                    row = next(row for row in rows if row["case"] == name)
                    valid = checkpoint_valid(case, row)
                    set_status(case, "watchdog-stopped", pid=process.pid, restartable=valid)
                    del active[name]
                    if valid and int(read_json(status_path(case)).get("restart_count", 0)) < args.max_restarts:
                        pending.append(row)
            if stop_requested:
                for process, case, log, _tracker in active.values():
                    os.killpg(process.pid, signal.SIGTERM)
                deadline = time.time() + args.term_grace_seconds
                for process, case, log, _tracker in active.values():
                    try: process.wait(timeout=max(deadline - time.time(), 0.0))
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait()
                    row = next(row for row in rows if row["case"] == case.name)
                    log.close(); set_status(case, "restartable" if checkpoint_valid(case, row) else "failed", pid=process.pid)
                break
            while pending and len(active) < min(args.max_jobs, 2):
                row = pending.pop(0); case = root / row["case"]; case.mkdir(exist_ok=True)
                state = classify(case, row)
                if args.skip_finished and state in TERMINAL:
                    set_status(case, state, skipped=True); continue
                if state in {"failed", "blocked-before-launch"}:
                    set_status(case, state, skipped=True); continue
                restarting = state in {"restartable", "watchdog-stopped"}
                if free_gib(root) < args.minimum_free_gib:
                    set_status(case, "blocked-before-launch", blocked_reason="minimum_free_space")
                    pending.insert(0, row); break
                count = int(read_json(status_path(case)).get("restart_count", 0)) + int(restarting)
                set_status(case, "running", restart_count=count, started_unix_s=time.time())
                log = (case / "run.log").open("a")
                process = subprocess.Popen(worker_command(row, case, args), cwd=Path(__file__).parents[1], env=worker_environment(row, case, args, restarting), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                set_status(case, "running", pid=process.pid, restart_count=count)
                active[row["case"]] = (process, case, log, {"physical_timestamp": time.time(), "liveness_timestamp": time.time()})
                maximum_active_observed = max(maximum_active_observed, len(active))
            atomic_json(root / "active_workers.json", {name: proc.pid for name, (proc, _case, _log, _tracker) in active.items()})
            if pending and not active and free_gib(root) < args.minimum_free_gib:
                break
            time.sleep(args.poll_seconds)
    finally:
        atomic_json(root / "qualification_supervisor_summary.json", {
            "maximum_active_observed": maximum_active_observed,
            "maximum_jobs_configured": min(args.max_jobs, 2),
            "finished_unix_s": time.time(),
        })
        (root / "active_workers.json").unlink(missing_ok=True)
        (root / "launcher.json").unlink(missing_ok=True)
        release_lock(root, ownership["token"])
    if not args.smoke_worker:
        completed = [
            artifacts(root / row["case"]) for row in matrix()
            if (artifacts(root / row["case"]) / "developed_fatigue_growth_summary.json").is_file()
        ]
        if completed:
            subprocess.run(
                [sys.executable, "scripts/analyze_v10_2_30_four_class_fatigue_campaign.py",
                 *(str(path) for path in completed), "--out", str(root / "analysis")],
                cwd=Path(__file__).parents[1], check=False,
            )
    return 0


def monitor(root: Path) -> int:
    print(f"disk_free_GiB={free_gib(root):.2f}")
    for row in matrix():
        case = root / row["case"]; payload = read_json(status_path(case)); payload.update(progress(case))
        print(row["case"], json.dumps({k: payload.get(k) for k in ("status", "pid", "cycles_reached", "event_count", "crack_extension_um", "current_mode", "current_phase", "latest_physical_progress_timestamp", "latest_liveness_timestamp", "restart_count")}, sort_keys=True))
    return 0


def stop_launcher(root: Path) -> int:
    launcher = read_json(root / "launcher.json")
    if launcher.get("pid"): os.kill(int(launcher["pid"]), signal.SIGTERM)
    return 0


def smoke_worker(case: Path) -> int:
    state = read_json(case / "smoke_checkpoint.json", {"step": 0})
    start = int(state.get("step", 0))
    for step in range(start + 1, 5):
        now = time.time()
        atomic_json(case / "smoke_checkpoint.json", {"step": step})
        atomic_json(case / "high_cycle_live_checkpoint.json", {"schema": "smoke", "timestamp_unix_s": now, "cycles_from_engine_time": step})
        (case / "high_cycle_live_state.npz").touch()
        from arrhenius_fracture.run_state_checkpoint_v10230 import write_combined_checkpoint
        threshold = 2.0
        action = 0.5
        write_combined_checkpoint(
            case,
            outer={
                "case": {"da_phys_m": 5e-6},
                "cycles_total": float(step),
                "geometry": {
                    "crack_tip_m": [0.001, 0.0],
                    "front_paths": [[[0.001, 0.0]]],
                    "front_inventory": [{"xy": [0.001, 0.0], "fwd": [1.0, 0.0],
                        "last_plane": {"t": [1.0, 0.0], "n": [0.0, 1.0]},
                        "win_plane": {"t": [1.0, 0.0], "n": [0.0, 1.0]}}],
                    "committed_event_count": 0,
                    "kinetic_event_index": 0,
                    "transaction_index": 0,
                    "transaction_state": "committed",
                    "mesh_metadata": {"hbar_m": 1.0, "hbar_tip_m": 1.0,
                                      "tip_reference_centers_m": [[0.001, 0.0]]},
                },
            },
            arrays={"smoke": [step]},
            kinetic={
                "geometry_signature": [0, 0.0, 0.0, 0.0],
                "stochastic": {
                    "B": action / threshold,
                    "hazard_threshold_action": threshold,
                    "hazard_action_current": action,
                    "hazard_event_index": 0,
                    "hazard_threshold_history": [],
                    "avalanche_base_checkpoint_m": 5e-6,
                    "avalanche_event_length_factor": 1.0,
                    "avalanche_event_advance_m": 5e-6,
                    "rng_state": {"smoke": step},
                },
            },
            kinetic_vector=[float(step)],
        )
        if os.environ.get("SMOKE_INTERRUPT") == "1" and step == 2: return 75
        time.sleep(float(os.environ.get("SMOKE_STEP_SECONDS", "0.05")))
    (case / "exit_code.txt").write_text("0\n")
    atomic_json(case / "v10_2_30_fixed_deltaK_control.json", {
        "censor_status": "propagated",
        "parameter_option": os.environ.get("PARAMETER_OPTION"),
        "target_deltaK_MPa_sqrt_m": float(os.environ.get("TARGET_DELTAK", "0")),
        "cleavage_hazard_seed": int(os.environ.get("HAZARD_SEED", "0")),
    })
    atomic_json(case / "developed_fatigue_growth_summary.json", {
        "status": "growth_target_reached", "event_count": 1,
        "cycles_consumed": 4.0, "final_projected_extension_um": 5.0,
        "target_reached": True,
        "developed_interval": {"event_count": 1, "da_dN": 1.25e-6},
        "provenance": {
            "parameter_option": os.environ.get("PARAMETER_OPTION"),
            "deltaK_MPa_sqrt_m": float(os.environ.get("TARGET_DELTAK", "0")),
            "hazard_seed": int(os.environ.get("HAZARD_SEED", "0")),
            "git_head": "smoke",
        },
    })
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="command", required=True)
    runp = sub.add_parser("run"); runp.add_argument("root", type=Path)
    runp.add_argument("--max-jobs", type=int, default=2); runp.add_argument("--skip-finished", action="store_true", default=True)
    runp.add_argument("--no-progress-seconds", type=float, default=float(os.environ.get("V10230_QUAL_NO_PROGRESS_SECONDS", "900")))
    runp.add_argument("--term-grace-seconds", type=float, default=30); runp.add_argument("--minimum-free-gib", type=float, default=float(os.environ.get("V10230_QUAL_MIN_FREE_GIB", "10")))
    runp.add_argument("--poll-seconds", type=float, default=2); runp.add_argument("--target-extension-um", type=float, default=25); runp.add_argument("--cycles-max", type=float, default=1e12)
    runp.add_argument("--smoke-worker", action="store_true")
    runp.add_argument("--max-restarts", type=int, default=3)
    runp.add_argument("--family-descriptor", type=Path, default=Path(__file__).parents[1] / "runtime_inputs/v10_2_30/qualification_family_manifest.json")
    runp.add_argument("--recover-stale-lock", action="store_true")
    mon = sub.add_parser("monitor"); mon.add_argument("root", type=Path)
    stp = sub.add_parser("stop"); stp.add_argument("root", type=Path)
    smoke = sub.add_parser("smoke-worker"); smoke.add_argument("case", type=Path)
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.command == "run": return run(args)
    if args.command == "monitor": return monitor(args.root.resolve())
    if args.command == "stop": return stop_launcher(args.root.resolve())
    return smoke_worker(args.case.resolve())


if __name__ == "__main__": raise SystemExit(main())
