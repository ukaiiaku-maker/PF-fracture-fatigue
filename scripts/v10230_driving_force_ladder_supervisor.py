#!/usr/bin/env python3
"""Stage and supervise the Peak/DBTT v10.2.30 high-driving-force ladder.

This module only specializes the qualified supervisor's launch matrix.  It does
not alter constitutive, stochastic, event-length, energy-gate, or localization
settings.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

try:
    from scripts import v10230_qualification_supervisor as q
except ModuleNotFoundError:
    import v10230_qualification_supervisor as q


FRACTIONS = (0.950, 0.975, 1.000)
LABELS = ("peak", "dbtt")
R = 0.1
FREQUENCY_HZ = 1000.0
TEMPERATURE_K = 300.0
TARGET_EXTENSION_UM = 100.0
CYCLES_MAX = 1e12
MANIFEST_NAME = "driving_force_ladder_matrix.json"


def case_name(label: str, fraction: float, seed: int) -> str:
    token = f"{fraction:.3f}".replace(".", "p")
    return f"{label}_f{token}_seed{seed}"


def matrix() -> list[dict]:
    rows = []
    for fraction in FRACTIONS:
        for label in LABELS:
            option, seed, reference = q.OPTIONS[label]
            delta_k = reference * fraction
            kmax = delta_k / (1.0 - R)
            rows.append({
                "case": case_name(label, fraction, seed),
                "label": label,
                "parameter_option": option,
                "seed": seed,
                "fraction": fraction,
                "reference_deltaK_MPa_sqrt_m": reference,
                "deltaK_MPa_sqrt_m": delta_k,
                "Kmax_MPa_sqrt_m": kmax,
                "Kmin_MPa_sqrt_m": R * kmax,
                "R": R,
                "frequency_Hz": FREQUENCY_HZ,
                "temperature_K": TEMPERATURE_K,
                "target_extension_um": TARGET_EXTENSION_UM,
                "cycle_horizon": CYCLES_MAX,
            })
    return rows


def prepare(root: Path, minimum_free_gib: float = 10.0) -> dict:
    root = root.resolve()
    repository = Path(__file__).resolve().parents[1]
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repository, text=True
    ).strip()
    if dirty:
        raise RuntimeError("clean worktree required")
    if root.exists():
        raise RuntimeError(f"destination already exists: {root}")
    free = q.free_gib(root.parent)
    if free < minimum_free_gib:
        raise RuntimeError(
            f"minimum free-space preflight failed: {free:.3f} < {minimum_free_gib:.3f} GiB"
        )
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=repository, text=True
    ).strip()
    rows = matrix()
    payload = {
        "schema": "v10.2.30_driving_force_ladder_v1",
        "branch": branch,
        "launch_git_head": head,
        "minimum_free_gib": minimum_free_gib,
        "available_free_gib": free,
        "maximum_concurrency": 2,
        "cases": rows,
    }
    root.mkdir(parents=True)
    for row in rows:
        case = root / row["case"]
        case.mkdir()
        q.set_status(case, "pending", pid=None, restart_count=0)
    q.atomic_json(root / MANIFEST_NAME, payload)
    return payload


def validate_staged(root: Path) -> dict:
    payload = q.read_json(root / MANIFEST_NAME)
    if payload.get("schema") != "v10.2.30_driving_force_ladder_v1":
        raise RuntimeError("invalid or missing driving-force ladder manifest")
    expected = matrix()
    if payload.get("cases") != expected:
        raise RuntimeError("staged driving-force ladder differs from committed matrix")
    return payload


def run(root: Path, args) -> int:
    validate_staged(root)
    q.matrix = matrix
    q.EXPECTED_MATRIX = {
        (row["label"], row["fraction"]): row["deltaK_MPa_sqrt_m"]
        for row in matrix()
    }
    argv = [
        "run", str(root), "--max-jobs", "2",
        "--target-extension-um", str(TARGET_EXTENSION_UM),
        "--cycles-max", str(CYCLES_MAX),
        "--minimum-free-gib", str(args.minimum_free_gib),
        "--no-progress-seconds", str(args.no_progress_seconds),
    ]
    if args.recover_stale_lock:
        argv.append("--recover-stale-lock")
    return q.run(q.parser().parse_args(argv))


def monitor(root: Path) -> int:
    print(f"disk_free_GiB={q.free_gib(root):.2f}")
    for row in matrix():
        case = root / row["case"]
        status = q.read_json(case / "qualification_status.json")
        status.update(q.progress(case))
        fields = ("status", "pid", "cycles_reached", "event_count",
                  "crack_extension_um", "current_mode", "restart_count")
        print(row["case"], json.dumps({k: status.get(k) for k in fields}, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("root", type=Path)
    prep.add_argument("--minimum-free-gib", type=float, default=10.0)
    runp = sub.add_parser("run")
    runp.add_argument("root", type=Path)
    runp.add_argument("--minimum-free-gib", type=float, default=10.0)
    runp.add_argument("--no-progress-seconds", type=float, default=900.0)
    runp.add_argument("--recover-stale-lock", action="store_true")
    for command in ("monitor", "stop"):
        sub.add_parser(command).add_argument("root", type=Path)
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    root = args.root.resolve()
    if args.command == "prepare":
        print(json.dumps(prepare(root, args.minimum_free_gib), indent=2, sort_keys=True))
        return 0
    if args.command == "run":
        return run(root, args)
    if args.command == "monitor":
        return monitor(root)
    return q.stop_launcher(root)


if __name__ == "__main__":
    raise SystemExit(main())
