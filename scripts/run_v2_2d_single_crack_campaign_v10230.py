#!/usr/bin/env python3
"""Bounded two-worker launcher for the named V2 2-D single-crack campaign."""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arrhenius_fracture.v2_named_parameterizations import load_exact_candidate, load_for

ALIASES = ("DBTT_V2", "DBTT_V2_P40", "weakT_V2_P25", "weakT_V2_P40", "ceramic_V2_P25", "ceramic_V2_P40")
TEMPERATURES = (300, 1200)
EXPECTED = {
    "DBTT_V2": ("P25_TJBSV2_S_002987", "8a16415705c47c7708ec35385cde6c9a53bb6b9de6edf98b902a1f40626a9f72"),
    "DBTT_V2_P40": ("P40_TJBSV2_S_038503", "be0343fd982eda89f7f7240bf599298a0e07e06eea87570ee34fa9935dabcbfa"),
    "weakT_V2_P25": ("P25_TJBSV2_S_026704", "85f8d9d2ddbe1e7c7c0c34589883b886023507089d0fe423c1ce2c4eb60d0041"),
    "weakT_V2_P40": ("P40_TJBSV2_S_038278", "98908d7eecb5916a86dc886900b6e4fd687f4d67daa2bd77e4d75ebaef06b3d6"),
    "ceramic_V2_P25": ("P25_TJBSV2_S_004127", "c794af3a2912fa91d587d06ae9177ba87af1da6d30e4ad7310d5f673edf741e3"),
    "ceramic_V2_P40": ("P40_TJBSV2_S_031027", "d76e6a011ba83f762b5830c03c75571c851936677d213cfa1fb600a00a286cfa"),
}
SEED = 3621


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def preflight(output: Path, python: Path, *, allow_resume: bool = False) -> list[dict]:
    if git("status", "--porcelain"):
        raise SystemExit("execution tree is dirty")
    if output.exists() and not allow_resume:
        raise SystemExit(f"fresh campaign output root must be absent: {output}")
    if shutil.disk_usage(output.parent).free < 1300 * 1024 * 1024:
        raise SystemExit("Data-volume free-space reserve is below 1.3 GiB")
    rows = []
    for alias in ALIASES:
        record = load_for(alias, "PF_sharp_front")
        exact = load_exact_candidate(record.source_candidate_id)
        candidate, digest = EXPECTED[alias]
        if (record.source_candidate_id, record.complete_bound_row_sha256) != (candidate, digest):
            raise SystemExit(f"frozen identity mismatch for {alias}")
        if dict(record.full_precision_row) != dict(exact.full_precision_row):
            raise SystemExit(f"alias/exact byte-parity failure for {alias}")
        rows.append({"alias": alias, "candidate_id": candidate, "row_sha256": digest})
    running = subprocess.run(["pgrep", "-fal", "arrhenius_fracture.*sharp_front"], text=True, capture_output=True)
    if running.returncode == 0 and running.stdout.strip():
        raise SystemExit("an existing heavy sharp-front worker is active; refusing to exceed the system-wide bound")
    if not python.is_file(): raise SystemExit(f"Python executable missing: {python}")
    return rows


def resolve_kernel(output_parent: Path, python: Path) -> Path:
    # Reuse the repository's audited direct-kernel cache.  Raw campaign data
    # still goes to the separately frozen Data-volume output root.
    cache = Path(os.environ.get("V10230_KERNEL_CACHE_ROOT", ROOT / "runs/v10_2_28_kernel_cache")).resolve()
    command = [str(python), str(ROOT / "scripts/ensure_v10_2_28_signed_kernel.py"),
        "--theta-deg", "0", "--target-extension-um", "1000", "--branching-mode", "single_front",
        "--maximum-fronts", "1", "--process-zone-length-um", "50", "--process-zone-bins", "80",
        "--mesh-nx", "36", "--mesh-ny", "72", "--tip-h-fine-um", "1", "--tip-ratio", "1.20",
        "--da-phys-um", "5", "--mode", "auto", "--cache-root", str(cache)]
    environment = os.environ.copy()
    environment.update({"PYTHON_BIN": str(python),
                        "CONDA_ENV": "arrhenius-sharp-front-v10-codex",
                        "CONDA_DEFAULT_ENV": "arrhenius-sharp-front-v10-codex"})
    result = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True)
    if result.returncode: raise SystemExit(result.stdout + result.stderr)
    family = Path(result.stdout.strip().splitlines()[-1]).resolve()
    if not family.is_file(): raise SystemExit("kernel resolver did not return a file")
    return family


def manifest(output: Path, python: Path, family: Path, rows: list[dict]) -> None:
    head, tree = git("rev-parse", "HEAD"), git("rev-parse", "HEAD^{tree}")
    payload = {
        "schema": "v10.2.30_v2_named_single_crack_campaign_v1",
        "classification": "V2_NAMED_PARAMETERIZATION_SINGLE_CRACK_SPATIAL_TRANSFER_TEST",
        "git_head": head, "git_tree": tree,
        "source_parent_commit": "c1b3f08d956242844ee1206bf10576f08f8a2859",
        "accepted_v2_source_commit": "6a81911e4fa208fdccad83ca7bde40c4d7b975c4",
        "python_executable": str(python), "python_sha256": sha(python),
        "launcher": str(Path(__file__).resolve()), "launcher_sha256": sha(Path(__file__)),
        "entrypoint": "arrhenius_fracture.sharp_front_v10_2_30_v2_named_single_crack",
        "kernel_family": str(family), "kernel_family_sha256": sha(family),
        "aliases": rows, "temperatures_K": list(TEMPERATURES), "seed": SEED,
        "common_random_numbers": True, "case_count": 12, "theta_deg": 0.0,
        "dU_m": 2e-7, "dt_s": 8.4, "target_projected_extension_um": 1000.0,
        "maximum_fronts": 1, "branching": False, "field_export": "sparse_accepted_v1",
        "source_diff_from_parent": git("diff", "--name-status", "c1b3f08d956242844ee1206bf10576f08f8a2859..HEAD").splitlines(),
    }
    (output / "immutable_launch_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with (output / "case_matrix.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", "alias", "candidate_id", "row_sha256", "temperature_K", "seed"]); w.writeheader()
        by = {r["alias"]: r for r in rows}
        for alias in ALIASES:
            for temp in TEMPERATURES:
                w.writerow({"case_id": f"{alias}_{temp}K_theta0", **by[alias], "temperature_K": temp, "seed": SEED})


def command(case: Path, alias: str, temp: int, python: Path, family: Path, steps: int) -> list[str]:
    return [str(python), "-u", "-m", "arrhenius_fracture.sharp_front_v10_2_30_v2_named_single_crack",
        "--v2-parameter-alias", alias, "--signed-kernel-family", str(family), "--mode", "2d",
        "--temperatures", str(temp), "--steps", str(steps), "--nx", "36", "--ny", "72",
        "--dU", "2e-7", "--dt", "8.4", "--n-stagger", "2", "--tip-h-fine", "1e-6",
        "--tip-ratio", "1.20", "--da-phys", "5e-6", "--target-crack-extension-um", "1000",
        "--front-state-model", "moving_pz", "--tip-source-model", "continuum",
        "--tip-kinetics-mode", "moving_velocity", "--bulk-plasticity-mode", "tip_only",
        "--directional-j-mode", "root_signed", "--tip-plasticity", "--active-shielding",
        "--signed-active-shielding", "--mobile-shield-fraction", "0", "--no-wake-shielding",
        "--crystal-aniso", "--crystal-compete", "--crystal-theta-deg", "0",
        "--crystal-material", "w", "--j-decomposition", "cluster", "--max-fronts", "1",
        "--crack-backend", "sharp_wake", "--adaptive-events", "--adaptive-event-target", "0.15",
        "--print-every", "200", "--save-snapshots", "0", "--snapshot-cols", "1", "--out", str(case)]


def terminalize(case: Path, alias: str, temp: int, rc: int) -> dict:
    from arrhenius_fracture.sparse_accepted_field_export_v10230 import export_latest_checkpoint
    if (case / "run_state_checkpoint.json").is_file():
        export_latest_checkpoint(case, reason=f"terminal_process_exit_{rc}")
    extension = 0.0; step = 0
    csv_path = case / f"steps_{temp:04d}K.csv"
    if csv_path.is_file():
        a = np.genfromtxt(csv_path, delimiter=",", names=True)
        if getattr(a, "size", 0):
            row = a if a.ndim == 0 else a[-1]; extension = float(row["crack_extension_m"]); step = int(row["step"])
    reached = rc == 0 and extension >= 1000e-6 - 1e-12
    reason = "REACHED_1000UM" if reached else (f"PROCESS_EXIT_{rc}" if rc else "TARGET_NOT_REACHED")
    status = "V2_SINGLE_CRACK_2D_PF_REACHED_1000UM" if reached else f"V2_SINGLE_CRACK_2D_PF_STOPPED_FAIL_CLOSED_{reason}"
    payload = {"case_id": f"{alias}_{temp}K_theta0", "alias": alias, "temperature_K": temp,
               "exit_code": rc, "target_reached": reached, "terminal_reason": reason,
               "classification": status, "accepted_step": step, "projected_extension_m": extension,
               "finished_unix_s": time.time()}
    (case / "terminal_record.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


async def run_one(sem, output, alias, temp, python, family, steps):
    async with sem:
        case = output / f"{alias}_{temp}K_theta0"; case.mkdir(exist_ok=True)
        cmd = command(case, alias, temp, python, family, steps)
        (case / "command.json").write_text(json.dumps(cmd, indent=2) + "\n")
        env = os.environ.copy(); env.update({"PYTHONPATH": str(ROOT), "PARAMETER_CAMPAIGN": "1",
            "CLEAVAGE_HAZARD_MODE": "exponential", "CLEAVAGE_EVENT_LENGTH_MODE": "threshold_scaled",
            "CLEAVAGE_EVENT_MIN_FACTOR": "0.5", "CLEAVAGE_EVENT_MAX_FACTOR": "4.0",
            "CLEAVAGE_EVENT_SUBSEGMENT_FRACTION": "0.1", "CLEAVAGE_HAZARD_SEED": str(SEED),
            "ANISOTROPIC_TRANSPORT_MODE": "validated_scalar", "ANISOTROPIC_USE_AVALANCHE_BACKEND": "1",
            "ANISOTROPIC_EMISSION_ENABLED": "1", "PERSISTENT_SOURCE_MIN_WIDTH_UM": "0",
            "EXPECTED_HEAD": git("rev-parse", "HEAD"), "V10230_HIGH_CYCLE_CHECKPOINT_DIR": str(case),
            "V10230_SPARSE_FIELD_EXPORT_DIR": str(case)})
        if (case / "run_state_checkpoint.json").is_file() and not (case / "terminal_record.json").is_file():
            env["V10230_RESTART_CHECKPOINT_DIR"] = str(case)
        status = {"state": "RUNNING", "pid": None, "started_unix_s": time.time()}
        log = (case / "run.log").open("ab")
        proc = await asyncio.create_subprocess_exec(*cmd, cwd=ROOT, env=env, stdout=log, stderr=asyncio.subprocess.STDOUT)
        status["pid"] = proc.pid; (case / "worker_status.json").write_text(json.dumps(status, indent=2) + "\n")
        rc = await proc.wait(); log.close()
        result = terminalize(case, alias, temp, rc)
        (case / "worker_status.json").write_text(json.dumps({"state": "TERMINAL", "pid": proc.pid, "exit_code": rc}, indent=2) + "\n")
        return result


async def campaign(args, output, python, family):
    sem = asyncio.Semaphore(args.max_workers)
    tasks = []
    for alias in ALIASES:
        for temp in TEMPERATURES:
            case = output / f"{alias}_{temp}K_theta0"
            if (case / "terminal_record.json").is_file(): continue
            tasks.append(run_one(sem, output, alias, temp, python, family, args.steps))
    return await asyncio.gather(*tasks)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--python", type=Path, default=Path(sys.executable)); p.add_argument("--max-workers", type=int, default=2)
    p.add_argument("--steps", type=int, default=2_000_000); p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args(); output = args.output_root.resolve(); python = args.python.resolve()
    if args.max_workers not in (1, 2): raise SystemExit("--max-workers must be 1 or 2")
    rows = preflight(output, python, allow_resume=(output.exists() and not args.preflight_only)); family = resolve_kernel(output.parent, python)
    if args.preflight_only:
        print(json.dumps({"preflight": "PASS", "cases": 12, "kernel": str(family)}, indent=2)); return
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "immutable_launch_manifest.json").exists(): manifest(output, python, family, rows)
    results = asyncio.run(campaign(args, output, python, family))
    (output / "campaign_terminal_records.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    subprocess.check_call([str(python), str(ROOT / "scripts/report_v2_2d_single_crack_campaign_v10230.py"), "--campaign-root", str(output)], cwd=ROOT)


if __name__ == "__main__":
    main()
