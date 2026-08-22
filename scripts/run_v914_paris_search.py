#!/usr/bin/env python3
"""Launch fresh Paris-first v9.14 v7 candidate trajectories."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "artifacts/v914_paris_reparameterization"
DEFAULT_CONFIG = ROOT / "config/v914_paris_search_target.json"
DEFAULT_PHYSICS = ROOT / "artifacts/v914_calibrated_v7_transport_extension/calibrated_v7_transport_extension.json"
TERMINAL = {"growth_target_reached", "cycle_horizon_reached", "work_budget_exhausted"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slug(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def load_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    values = []
    for line in path.read_text().splitlines():
        value = line.strip().split(",", 1)[0]
        if value and value != "candidate_id":
            values.append(value)
    return set(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fidelity", choices=("level0", "level1", "level2", "level3"), required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_ARTIFACT / "candidate_registry.csv")
    parser.add_argument("--physics", type=Path, default=DEFAULT_PHYSICS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--candidate-ids", type=Path)
    parser.add_argument(
        "--case-table",
        type=Path,
        help="Optional candidate-specific CSV containing candidate_id and DeltaK_MPa_sqrt_m.",
    )
    parser.add_argument(
        "--campaign-label",
        help="Optional output subdirectory label; the physical fidelity metadata is unchanged.",
    )
    parser.add_argument("--out-root", type=Path, default=ROOT / "runs/v914_paris_reparameterization_v1")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.worker_count < 1 or not 0 <= args.worker_index < args.worker_count:
        raise SystemExit("worker-index must lie in [0, worker-count)")

    config = json.loads(args.config.read_text())
    fidelity = dict(config[args.fidelity])
    if args.fidelity in {"level2", "level3"} and args.case_table is None and "DeltaK_MPa_sqrt_m" not in fidelity:
        raise SystemExit(f"{args.fidelity} requires an adaptive candidate/load table via --case-table")
    requested = load_ids(args.candidate_ids)
    with args.registry.open(newline="") as stream:
        registry = list(csv.DictReader(stream))
    if requested is not None:
        registry = [row for row in registry if row["candidate_id"] in requested]
        missing = requested - {row["candidate_id"] for row in registry}
        if missing:
            raise SystemExit(f"candidate IDs absent from registry: {sorted(missing)}")
    if not registry:
        raise SystemExit("no candidates selected")

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if dirty and not args.dry_run:
        raise SystemExit("production Paris search requires a clean worktree")
    if args.case_table is not None:
        with args.case_table.open(newline="") as stream:
            requested_cases = list(csv.DictReader(stream))
        registry_by_id = {row["candidate_id"]: row for row in registry}
        unknown = {row["candidate_id"] for row in requested_cases} - set(registry_by_id)
        if unknown:
            raise SystemExit(f"case-table candidates absent from registry: {sorted(unknown)}")
        pairs = [(registry_by_id[row["candidate_id"]], row) for row in requested_cases]
    else:
        pairs = [(row, {"DeltaK_MPa_sqrt_m": value}) for row in registry for value in fidelity["DeltaK_MPa_sqrt_m"]]
    cases = [
        {
            "case_index": index,
            "candidate_id": row["candidate_id"],
            "source_historical_sobol_index": row["source_historical_sobol_index"],
            "paris_design_role": row["paris_design_role"],
            "fidelity": args.fidelity,
            "deltaK_MPa_sqrt_m": float(case["DeltaK_MPa_sqrt_m"]),
            "R": float(case.get("R") or fidelity["R"]),
            "frequency_Hz": float(case.get("frequency_Hz") or fidelity["frequency_Hz"]),
            "temperature_K": float(case.get("temperature_K") or fidelity["temperature_K"]),
            "seed": int(case.get("seed") or fidelity["seed"]),
        }
        for index, (row, case) in enumerate(pairs)
    ]
    output = args.out_root / (args.campaign_label or args.fidelity)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "v914_paris_search_launch_manifest_v1",
        "fidelity": args.fidelity,
        "repository_head": head,
        "fresh_uninterrupted_required": True,
        "restart_supported": False,
        "candidate_registry": str(args.registry.resolve()),
        "candidate_registry_sha256": digest(args.registry),
        "fixed_current_physics": str(args.physics.resolve()),
        "fixed_current_physics_sha256": digest(args.physics),
        "external_target_config": str(args.config.resolve()),
        "external_target_config_sha256": digest(args.config),
        "candidate_specific_case_table": str(args.case_table.resolve()) if args.case_table else None,
        "candidate_specific_case_table_sha256": digest(args.case_table) if args.case_table else None,
        "numerical_fidelity": fidelity,
        "cases": cases,
    }
    (output / "launch_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with (output / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)

    selected = [case for case in cases if int(case["case_index"]) % args.worker_count == args.worker_index]
    for sequence, case in enumerate(selected, 1):
        run_dir = output / case["candidate_id"] / f"DeltaK{slug(case['deltaK_MPa_sqrt_m'])}_seed{case['seed']}"
        result_path = run_dir / "result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            if result.get("status") in TERMINAL and int(result.get("restart_count", -1)) == 0:
                print(f"SKIP accepted fresh terminal {run_dir}", flush=True)
                continue
            raise SystemExit(f"refusing to overwrite invalid prior result {run_dir}")
        if (run_dir / "checkpoint.json").exists():
            raise SystemExit(f"refusing to resume interrupted trajectory {run_dir}")
        command = [
            sys.executable,
            str(ROOT / "scripts/run_v914_v7_vhcf_event_to_event.py"),
            "--registry", str(args.registry.resolve()),
            "--physics", str(args.physics.resolve()),
            "--candidate", case["candidate_id"],
            "--deltaK", str(case["deltaK_MPa_sqrt_m"]),
            "--R", str(case["R"]),
            "--return-control", "REV",
            "--configuration-label", "CAL-TRANS-REV",
            "--frequency-Hz", str(case["frequency_Hz"]),
            "--temperature-K", str(case["temperature_K"]),
            "--seed", str(case["seed"]),
            "--n-bins", str(fidelity["n_bins"]),
            "--base-phase-intervals", str(fidelity["base_phase_intervals"]),
            "--maximum-physical-cycles", str(fidelity["maximum_physical_cycles"]),
            "--maximum-cycle-map-evaluations", str(fidelity.get("maximum_cycle_map_evaluations", 4096)),
            "--target-extension-um", str(fidelity["target_extension_um"]),
            "--expected-head", head,
            "--out", str(run_dir),
        ]
        print(f"WORKER {args.worker_index} CASE {sequence}/{len(selected)}: {' '.join(command)}", flush=True)
        if not args.dry_run:
            run_dir.mkdir(parents=True, exist_ok=False)
            subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
