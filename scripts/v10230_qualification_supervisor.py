#!/usr/bin/env python3
"""Restart-safe supervisor for the bounded v10.2.30 four-class qualification."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


OPTIONS = {
    "peak": ("v913_paper_peak01_0242980_persistent_sites", 1720, 21.289546465050222),
    "dbtt": ("v913_paper_dbtt01_0202500_persistent_sites", 1001723, 21.02530765128298),
    "weakT": ("v913_paper_weakT01_0129902_persistent_sites", 2001726, 12.702935563752424),
    "ceramic": ("v913_paper_ceramic01_0077080_persistent_sites", 3001729, 12.259477791864454),
}
FRACTIONS = (0.55, 0.75, 0.95)
TERMINAL = {"completed", "censored"}


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


def checkpoint_valid(case: Path) -> bool:
    case = artifacts(case)
    payload = read_json(case / "high_cycle_live_checkpoint.json")
    return bool(
        payload.get("schema")
        and (case / "high_cycle_live_state.npz").is_file()
        and payload.get("timestamp_unix_s")
    )


def progress(case: Path) -> dict:
    case = artifacts(case)
    checkpoint = read_json(case / "high_cycle_live_checkpoint.json")
    summary = read_json(case / "developed_fatigue_growth_summary.json")
    geometry = read_json(case / "stochastic_avalanche_geometry_events.json", [])
    candidates = [p for p in (case / "high_cycle_live_checkpoint.json", case / "run.log") if p.exists()]
    stamp = max((p.stat().st_mtime for p in candidates), default=0.0)
    return {
        "cycles_reached": checkpoint.get("cycles_from_engine_time", summary.get("cycles_consumed", 0.0)),
        "event_count": len(geometry) if isinstance(geometry, list) else summary.get("event_count", 0),
        "crack_extension_um": summary.get("final_projected_extension_um", 0.0),
        "latest_progress_timestamp": stamp,
        "checkpoint_valid": checkpoint_valid(case),
    }


def status_path(case: Path) -> Path:
    return case / "qualification_status.json"


def set_status(case: Path, state: str, **extra) -> dict:
    old = read_json(status_path(case))
    payload = {**old, **progress(case), **extra, "status": state, "updated_unix_s": time.time()}
    atomic_json(status_path(case), payload)
    return payload


def classify(case: Path) -> str:
    old = read_json(status_path(case))
    if old.get("status") in TERMINAL:
        return old["status"]
    output = artifacts(case)
    if (output / "exit_code.txt").is_file():
        try:
            code = int((output / "exit_code.txt").read_text().strip())
        except ValueError:
            code = 1
        control = read_json(output / "v10_2_30_fixed_deltaK_control.json")
        if code == 0 and control.get("censor_status") == "right_censored_no_event":
            return "censored"
        if code == 0:
            return "completed"
        return "restartable" if checkpoint_valid(case) else "failed"
    return "restartable" if checkpoint_valid(case) else "pending"


def stale(case: Path, now: float, interval: float) -> bool:
    stamp = float(progress(case)["latest_progress_timestamp"] or 0.0)
    return stamp > 0.0 and now - stamp > interval


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
    })
    if restarting:
        env["V10230_RESTART_CHECKPOINT_DIR"] = str(case)
    return env


def run(args) -> int:
    root = args.root.resolve(); root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "qualification_matrix.json", {"cases": matrix()})
    atomic_json(root / "launcher.json", {"pid": os.getpid(), "started_unix_s": time.time()})
    for row in matrix():
        case = root / row["case"]; case.mkdir(exist_ok=True)
        if not status_path(case).exists():
            set_status(case, "pending", pid=None, restart_count=0)
    stop_requested = False
    def stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    active: dict[str, tuple[subprocess.Popen, Path, object]] = {}
    maximum_active_observed = 0
    pending = list(matrix())
    try:
        while pending or active:
            for name, (process, case, log) in list(active.items()):
                state = None
                if process.poll() is not None:
                    log.close(); state = classify(case)
                    if state not in TERMINAL:
                        state = "restartable" if checkpoint_valid(case) else "failed"
                    set_status(case, state, pid=process.pid, exit_code=process.returncode)
                    del active[name]
                elif stale(case, time.time(), args.no_progress_seconds):
                    os.killpg(process.pid, signal.SIGTERM)
                    try: process.wait(timeout=args.term_grace_seconds)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait()
                    log.close()
                    set_status(case, "watchdog-stopped", pid=process.pid, restartable=checkpoint_valid(case))
                    del active[name]
            if stop_requested:
                for process, case, log in active.values():
                    os.killpg(process.pid, signal.SIGTERM)
                deadline = time.time() + args.term_grace_seconds
                for process, case, log in active.values():
                    try: process.wait(timeout=max(deadline - time.time(), 0.0))
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait()
                    log.close(); set_status(case, "restartable" if checkpoint_valid(case) else "failed", pid=process.pid)
                break
            while pending and len(active) < min(args.max_jobs, 2):
                row = pending.pop(0); case = root / row["case"]; case.mkdir(exist_ok=True)
                state = classify(case)
                if args.skip_finished and state in TERMINAL:
                    set_status(case, state, skipped=True); continue
                restarting = state in {"restartable", "watchdog-stopped"}
                if restarting and not args.smoke_worker and not args.production_resume_supported:
                    set_status(case, "restartable", blocked_reason="outer_geometry_resume_not_supported")
                    continue
                if free_gib(root) < args.minimum_free_gib:
                    set_status(case, "pending", blocked_reason="minimum_free_space")
                    pending.insert(0, row); break
                count = int(read_json(status_path(case)).get("restart_count", 0)) + int(restarting)
                set_status(case, "running", restart_count=count, started_unix_s=time.time())
                log = (case / "run.log").open("a")
                process = subprocess.Popen(worker_command(row, case, args), cwd=Path(__file__).parents[1], env=worker_environment(row, case, args, restarting), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                set_status(case, "running", pid=process.pid, restart_count=count)
                active[row["case"]] = (process, case, log)
                maximum_active_observed = max(maximum_active_observed, len(active))
            atomic_json(root / "active_workers.json", {name: proc.pid for name, (proc, _case, _log) in active.items()})
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
        print(row["case"], json.dumps({k: payload.get(k) for k in ("status", "pid", "cycles_reached", "event_count", "crack_extension_um", "latest_progress_timestamp", "restart_count")}, sort_keys=True))
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
    runp.add_argument("--no-progress-seconds", type=float, default=float(os.environ.get("V10230_QUAL_NO_PROGRESS_SECONDS", "300")))
    runp.add_argument("--term-grace-seconds", type=float, default=30); runp.add_argument("--minimum-free-gib", type=float, default=float(os.environ.get("V10230_QUAL_MIN_FREE_GIB", "10")))
    runp.add_argument("--poll-seconds", type=float, default=2); runp.add_argument("--target-extension-um", type=float, default=25); runp.add_argument("--cycles-max", type=float, default=1e12)
    runp.add_argument("--smoke-worker", action="store_true"); runp.add_argument("--production-resume-supported", action="store_true")
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
