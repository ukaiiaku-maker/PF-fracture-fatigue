#!/usr/bin/env python3
"""Matched evolving/frozen-plastic fracture controls for finalist attribution."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.cohesive_arrhenius import COHESIVE_KINETIC_CZM
from arrhenius_fracture.coupled_opening_reduction_r10 import coupled_monotonic_ramp
from arrhenius_fracture.current_coupled_execution_r4 import R4_KERNEL_FAMILY
from arrhenius_fracture.retained_material_cohesive_r10 import (
    build_cohesive_current_engine, load_cohesive_binding,
)
from scripts.run_r10_coupled_opening_search import arm
from scripts.run_r10_opening_alternatives import mechanics, thermal_from


def atomic(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def execute(task):
    registry = Path(task["registry"])
    row = task["row"]
    cohesive = json.loads(row["cohesive_parameter_vector"])
    thermal = thermal_from(cohesive)
    _, stiffness = mechanics(thermal, task["temperature_K"])
    results = {}
    for label, feedback in (("evolving", True), ("frozen", False)):
        binding = load_cohesive_binding(
            registry, row["candidate_id"], execution_identity=COHESIVE_KINETIC_CZM)
        engine = build_cohesive_current_engine(
            binding, kernel_family_path=R4_KERNEL_FAMILY)
        results[label] = coupled_monotonic_ramp(
            engine, arm(row, cohesive), thermal, stiffness, task["temperature_K"],
            task["Kdot"], task["dK"], task["Kmax"], plastic_feedback=feedback)
    return task, {
        "schema": "r10.12h-matched-plastic-feedback-fracture-control/1",
        "candidate_id": row["candidate_id"],
        "source_material_candidate_id": row["source_material_candidate_id"],
        "complete_row_sha256": row["complete_row_sha256"],
        "plasticity_physics_sha256": row["plasticity_physics_sha256"],
        "cohesive_parameter_sha256": row["cohesive_parameter_sha256"],
        "temperature_K": task["temperature_K"],
        "evolving_full_state": results["evolving"],
        "frozen_initial_plastic_state": results["frozen"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--candidate-ids", required=True)
    parser.add_argument("--temperatures", default="300,450,600,750,900,1050,1200")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--Kdot", type=float, default=.005)
    parser.add_argument("--dK", type=float, default=.1)
    parser.add_argument("--Kmax", type=float, default=200.)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args(argv)
    wanted = {x.strip() for x in args.candidate_ids.split(",") if x.strip()}
    temperatures = [float(x) for x in args.temperatures.split(",")]
    with args.registry.open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["candidate_id"] in wanted]
    if {r["candidate_id"] for r in rows} != wanted:
        raise SystemExit("one or more requested candidates are absent from the registry")
    tasks = [{"registry": str(args.registry), "row": row, "temperature_K": T,
              "Kdot": args.Kdot, "dK": args.dK, "Kmax": args.Kmax}
             for row in rows for T in temperatures]
    summaries = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute, task) for task in tasks]
        for future in as_completed(futures):
            task, result = future.result()
            path = (args.out / result["candidate_id"] /
                    f"T_{int(task['temperature_K']):04d}K.json")
            atomic(path, result)
            evolving = result["evolving_full_state"]
            frozen = result["frozen_initial_plastic_state"]
            row = {
                "candidate_id": result["candidate_id"],
                "temperature_K": result["temperature_K"],
                "evolving_commit_K_MPa_sqrt_m": evolving["first_committed_separation_K_MPa_sqrt_m"],
                "frozen_commit_K_MPa_sqrt_m": frozen["first_committed_separation_K_MPa_sqrt_m"],
                "evolving_status": evolving["status"],
                "frozen_status": frozen["status"],
                "evolving_unsupported": evolving["unsupported_crossover"],
                "frozen_unsupported": frozen["unsupported_crossover"],
                "evolving_terminal_K_MPa_sqrt_m": evolving["Kmax_MPa_sqrt_m"],
                "frozen_terminal_K_MPa_sqrt_m": frozen["Kmax_MPa_sqrt_m"],
            }
            summaries.append(row)
            print(json.dumps(row), flush=True)
    summaries.sort(key=lambda r: (r["candidate_id"], r["temperature_K"]))
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "mechanism_control_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(summaries[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
