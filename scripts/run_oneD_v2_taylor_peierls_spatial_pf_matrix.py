#!/usr/bin/env python3
"""Run only missing DBTT spatial-transfer PF cases with a two-worker cap."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRANSFER = Path(
    "/private/tmp/oneD-v2-taylor-peierls-rcurve-search/analysis_outputs/"
    "oneD_v2_taylor_peierls_spatial_transfer"
)
DEFAULT_OUT = Path("/private/tmp/oneD-v2-taylor-peierls-spatial-pf-runs")
EXISTING_CASES = {
    "v913_zeroD_sobol_0202500": Path(
        "/private/tmp/oneD-v2-taylor-peierls-pf-runs-300um/"
        "oneD_v2_DBTT_control/T1100K_seed1008666"
    ),
    "oneD_v2_dbtt_TP_6ca03f05fbae34e9": Path(
        "/private/tmp/oneD-v2-taylor-peierls-pf-runs-300um/"
        "oneD_v2_DBTT_TP/T1100K_seed1008666"
    ),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_case(path: Path, target_um: float) -> bool:
    steps = path / "steps_1100K.csv"
    audit = path / "anisotropic_emission_audit_v10174.json"
    if not steps.is_file() or not audit.is_file():
        return False
    frame = pd.read_csv(steps, usecols=["crack_extension_m"])
    return bool(len(frame) and frame.crack_extension_m.max() * 1.0e6 >= target_um - 1.0e-9)


def launch(option: str, case_out: Path, transfer: Path, target_um: float):
    case_out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "TEMPERATURE_K": "1100",
        "PARAMETER_OPTION": option,
        "HAZARD_SEED": "1008666",
        "CASE_OUT": str(case_out),
        "TRANSFER_ROOT": str(transfer),
        "TRANSFER_REGISTRY": str(transfer / "oneD_v2_spatial_transfer_pf_registry.csv"),
        "TRANSFER_SELECTION": str(transfer / "oneD_v2_spatial_transfer_pf_selection.json"),
        "TARGET_EXTENSION_UM": f"{target_um:.17g}",
    })
    log = (case_out / "run.log").open("w")
    process = subprocess.Popen(
        ["bash", str(ROOT / "scripts" / "run_oneD_v2_terminal_pf_case.sh")],
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    return process, log


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer-root", type=Path, default=DEFAULT_TRANSFER)
    parser.add_argument("--outroot", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-um", type=float, default=300.0)
    parser.add_argument("--maximum-workers", type=int, default=2)
    args = parser.parse_args()
    if args.maximum_workers != 2:
        raise SystemExit("spatial PF matrix requires exactly two maximum workers")
    transfer = args.transfer_root.resolve()
    registry_path = transfer / "oneD_v2_spatial_transfer_pf_registry.csv"
    selection_path = transfer / "oneD_v2_spatial_transfer_pf_selection.json"
    rows = list(csv.DictReader(registry_path.open(newline="")))
    selection = json.loads(selection_path.read_text())
    if len(rows) != 8 or len({row["candidate_id"] for row in rows}) != 8:
        raise SystemExit("spatial PF transfer requires eight unique candidates")
    if selection["maximum_concurrent_heavy_workers"] != 2:
        raise SystemExit("selection does not certify the two-worker cap")
    args.outroot.mkdir(parents=True, exist_ok=True)
    cases = []
    pending = []
    for row in rows:
        candidate = row["candidate_id"]
        option = row["option_key"]
        existing = EXISTING_CASES.get(candidate)
        if existing is not None and valid_case(existing, args.target_um):
            cases.append({
                "option_key": option,
                "candidate_id": candidate,
                "selection_role": row["role"],
                "case_path": str(existing),
                "execution_status": "REUSED_COMPATIBLE_EXISTING_300UM_CASE",
                "fresh_PF_run": False,
            })
            print(f"PF_REUSE option={option} candidate={candidate} path={existing}", flush=True)
            continue
        case_out = args.outroot / option / "T1100K_seed1008666"
        if valid_case(case_out, args.target_um):
            cases.append({
                "option_key": option,
                "candidate_id": candidate,
                "selection_role": row["role"],
                "case_path": str(case_out),
                "execution_status": "REUSED_COMPLETED_SPATIAL_TRANSFER_CASE",
                "fresh_PF_run": False,
            })
        else:
            pending.append((row, case_out))

    running = []
    while pending or running:
        while pending and len(running) < args.maximum_workers:
            row, case_out = pending.pop(0)
            process, log = launch(row["option_key"], case_out, transfer, args.target_um)
            running.append((row, case_out, process, log))
            print(
                f"PF_LAUNCH option={row['option_key']} candidate={row['candidate_id']} "
                f"active_workers={len(running)}",
                flush=True,
            )
        time.sleep(1.0)
        survivors = []
        for row, case_out, process, log in running:
            code = process.poll()
            if code is None:
                survivors.append((row, case_out, process, log))
                continue
            log.close()
            if code != 0 or not valid_case(case_out, args.target_um):
                raise SystemExit(
                    f"PF case failed or did not reach target: {row['option_key']} code={code}"
                )
            cases.append({
                "option_key": row["option_key"],
                "candidate_id": row["candidate_id"],
                "selection_role": row["role"],
                "case_path": str(case_out),
                "execution_status": "FRESH_SPATIAL_TRANSFER_CASE_COMPLETE",
                "fresh_PF_run": True,
            })
            print(f"PF_COMPLETE option={row['option_key']} active_workers={len(survivors)}", flush=True)
        running = survivors

    order = {row["option_key"]: index for index, row in enumerate(rows)}
    cases.sort(key=lambda row: order[row["option_key"]])
    manifest = {
        "schema": "oneD_v2_taylor_peierls_spatial_pf_matrix_v1",
        "temperature_K": 1100.0,
        "hazard_seed": 1008666,
        "target_extension_um": args.target_um,
        "theta_deg": 0.0,
        "bulk_plasticity_mode": "tip_only",
        "maximum_concurrent_heavy_workers": args.maximum_workers,
        "candidate_count": len(cases),
        "fresh_PF_run_count": sum(bool(row["fresh_PF_run"]) for row in cases),
        "reused_PF_run_count": sum(not bool(row["fresh_PF_run"]) for row in cases),
        "new_FEMCZM_runs": 0,
        "registry_sha256": sha(registry_path),
        "selection_sha256": sha(selection_path),
        "pf_runner_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "cases": cases,
    }
    (args.outroot / "spatial_pf_matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"PF_MATRIX_COMPLETE cases={len(cases)} fresh={manifest['fresh_PF_run_count']} "
        "maximum_workers=2",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
