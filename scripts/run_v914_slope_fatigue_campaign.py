#!/usr/bin/env python3
"""Launch the adaptive slope campaign through qualified production runners.

The wrapper changes no physics.  It pins every immutable input, limits worker
concurrency, refuses ambiguous partial outputs, and keeps accelerated and
explicit-cycle overlap results in disjoint directories.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V914 = Path("/Volumes/Data/Data/Nanopillar_calculation/Arrhenius_FEM_CZM_MPZ_v9_14_cyclic_fatigue_knee_search")
DEFAULT_PYTHON = Path("/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--loads", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("accelerated", "explicit"), required=True)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--v914-root", type=Path, default=DEFAULT_V914)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--minimum-free-gib", type=float, default=3.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bool_column(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"1", "true", "yes"})


def fraction_label(value: float) -> str:
    return f"{value:.8f}".rstrip("0").rstrip(".").replace(".", "p")


def accelerated_command(args: argparse.Namespace, candidate: str, fractions: list[float], out: Path) -> list[str]:
    return [
        str(args.python), "-u", str(args.v914_root / "scripts/run_v914_fatigue_campaign.py"),
        "--registry", str(args.registry), "--physics", str(args.physics), "--out", str(out),
        "--candidate", candidate, "--fractions", *[f"{value:.17g}" for value in fractions],
        "--seeds", "1720", "--R", "0.1", "--frequency-Hz", "1000",
        "--temperature-K", "300", "--maximum-cycles", "1e14",
        "--target-extension-um", "100", "--phase-steps", "32",
        "--maximum-explicit-cycles", "4096", "--dmd-burst-cycles", "24",
        "--checkpoint-wait-cycles", "1e9",
    ]


def explicit_command(args: argparse.Namespace, row, out: Path) -> list[str]:
    return [
        str(args.python), "-u", str(ROOT / "scripts/run_v1032_explicit_lcf.py"),
        "--registry", str(args.registry), "--physics", str(args.physics),
        "--candidate", str(row.candidate_id), "--deltaK", f"{float(row.deltaK_MPa_sqrt_m):.17g}",
        "--mode", "explicit", "--phase-steps", "32", "--target-um", "100",
        "--maximum-cycles", "20000", "--seed", "1720", "--normalized-f", f"{float(row.normalized_f):.17g}",
        "--expected-head", args.expected_head, "--checkpoint-cycle-interval", "10",
        "--state-history-cycle-interval", "10", "--out", str(out),
    ]


def run_task(command: list[str], out: Path, terminal_name: str, expected_results: int, env: dict[str, str]) -> dict[str, object]:
    terminal = out / terminal_name
    if terminal.exists():
        return {"out": str(out), "status": "already_terminal", "returncode": 0, "wall_seconds": 0.0}
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"nonterminal output requires audit: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (out / "launcher.log").open("w") as stream:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True)
    wall = time.monotonic() - started
    result_name = "fatigue_result.json" if terminal_name == "WRAPPER_COMPLETE.json" else terminal_name
    result_count = sum(1 for _ in out.rglob(result_name))
    if process.returncode or result_count != expected_results:
        raise RuntimeError(f"production task failed rc={process.returncode}: {out}")
    if terminal_name == "WRAPPER_COMPLETE.json":
        terminal.write_text(json.dumps({"schema": "v914_slope_accelerated_wrapper_terminal_v1", "result_count": result_count}, indent=2) + "\n")
    return {"out": str(out), "status": "terminal", "returncode": process.returncode, "wall_seconds": wall, "result_count": result_count}


def main() -> int:
    args = parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if head != args.expected_head or dirty:
        raise RuntimeError(f"launch requires exact clean HEAD {args.expected_head}; actual={head} dirty={bool(dirty)}")
    if args.jobs < 1 or args.jobs > 3:
        raise RuntimeError("jobs must lie in [1,3]")
    loads = pd.read_csv(args.loads)
    tasks: list[tuple[list[str], Path, str, int]] = []
    if args.mode == "accelerated":
        selected = loads[bool_column(loads.accelerated_required)]
        for candidate, group in selected.groupby("candidate_id"):
            out = args.out / "accelerated" / str(candidate)
            fractions = group.normalized_f.astype(float).tolist()
            tasks.append((accelerated_command(args, str(candidate), fractions, out), out, "WRAPPER_COMPLETE.json", len(fractions)))
    else:
        selected = loads[bool_column(loads.explicit_required)]
        for row in selected.itertuples(index=False):
            out = args.out / "explicit" / str(row.candidate_id) / f"f{fraction_label(float(row.normalized_f))}_explicit"
            tasks.append((explicit_command(args, row, out), out, "result.json", 1))
    free = shutil.disk_usage("/Volumes/Data").free / 2**30
    if free < args.minimum_free_gib:
        raise RuntimeError(f"disk safety gate: {free:.2f} GiB free")
    args.out.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema": "v914_prospective_slope_campaign_wrapper_v1",
        "mode": args.mode,
        "driver_git_head": head,
        "registry": str(args.registry.resolve()), "registry_sha256": sha256(args.registry),
        "physics": str(args.physics.resolve()), "physics_sha256": sha256(args.physics),
        "loads": str(args.loads.resolve()), "loads_sha256": sha256(args.loads),
        "external_runner_sha256": sha256(args.v914_root / "scripts/run_v914_fatigue_campaign.py"),
        "explicit_runner_sha256": sha256(ROOT / "scripts/run_v1032_explicit_lcf.py"),
        "task_count": len(tasks), "maximum_parallel_workers": args.jobs,
        "maximum_explicit_cycles_accelerated_runner": 4096,
        "fatigue_specific_refit": False, "physics_changed": False,
    }
    (args.out / f"{args.mode}_launch_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{args.v914_root}:{ROOT / 'scripts'}"
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {}
        for command, out, terminal, expected_results in tasks:
            free = shutil.disk_usage("/Volumes/Data").free / 2**30
            if free < args.minimum_free_gib:
                raise RuntimeError(f"disk safety gate while launching: {free:.2f} GiB free")
            futures[pool.submit(run_task, command, out, terminal, expected_results, env)] = out
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"V914_SLOPE_TASK_COMPLETE mode={args.mode} out={result['out']} wall_s={result['wall_seconds']:.3f}", flush=True)
    (args.out / f"{args.mode}_launch_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"V914_SLOPE_CAMPAIGN_COMPLETE mode={args.mode} tasks={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
