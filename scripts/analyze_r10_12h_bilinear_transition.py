#!/usr/bin/env python3
"""Resolve the stable-to-commitment bracket for selected bilinear mixed arms."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.cohesive_arrhenius import COHESIVE_KINETIC_CZM
from arrhenius_fracture.current_coupled_execution_r4 import R4_KERNEL_FAMILY
from arrhenius_fracture.czm_high_cycle_adapter_r10 import install_czm_owner, opening_snapshot
from arrhenius_fracture.opening_alternatives_r10 import (
    OpeningAlternative, frozen_commitment_density_reference,
)
from arrhenius_fracture.retained_material_cohesive_r10 import (
    build_cohesive_current_engine, load_cohesive_binding,
)
from scripts.run_r10_coupled_opening_search import reference_load
from scripts.run_r10_opening_alternatives import mechanics, thermal_from


def write(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--candidate-ids", required=True)
    parser.add_argument("--factors", default=".925,.93,.935,.94,.945,.95,.955,.96,.965,.97")
    parser.add_argument("--grids", default="65,129,257")
    parser.add_argument("--temperature-K", type=float, default=300.)
    parser.add_argument("--frequency-Hz", type=float, default=1000.)
    parser.add_argument("--phases", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    with args.registry.open(newline="") as stream:
        registry = {row["candidate_id"]: row for row in csv.DictReader(stream)}
    factors = [float(value) for value in args.factors.split(",")]
    grids = [int(value) for value in args.grids.split(",")]
    duration = 1. / args.frequency_Hz / args.phases
    output = []
    for seed, candidate_id in enumerate(args.candidate_ids.split(","), 1):
        row = registry[candidate_id]
        if row["selected_shape"] != "bilinear" or row["selected_mode"] != "cohesive_mixed":
            raise ValueError(f"{candidate_id} is not a bilinear mixed arm")
        cohesive = json.loads(row["cohesive_parameter_vector"])
        thermal = thermal_from(cohesive)
        _, kenv = mechanics(thermal, args.temperature_K)
        alternative = OpeningAlternative(
            "cohesive_mixed", "bilinear", "time_dependent_first_passage",
            float(cohesive["activation_area_m2"]), float(cohesive["attempt_frequency_s"]),
        )
        kref, _ = reference_load(row, args.temperature_K, args.phases)
        for factor in factors:
            engine = build_cohesive_current_engine(
                load_cohesive_binding(args.registry, candidate_id,
                                      execution_identity=COHESIVE_KINETIC_CZM),
                kernel_family_path=R4_KERNEL_FAMILY,
            )
            install_czm_owner(engine, alternative, thermal, kenv, args.phases, seed,
                              frequency_Hz=args.frequency_Hz)
            snapshot = opening_snapshot(engine, factor * kref * 1e6, args.temperature_K)
            opening = engine._r10_czm_last_opening
            probabilities = []
            means = []
            if opening["kinetic_validity"] == "STATIC_SADDLE_MFPT_HAZARD_APPROXIMATION":
                for grid in grids:
                    reference = frozen_commitment_density_reference(
                        alternative, snapshot["environment_imposed_opening_m"], kenv,
                        thermal, args.temperature_K, opening, duration, grid,
                    )
                    probabilities.append(float(reference["committed_probability"]))
                    means.append(reference["mean_commitment_time_s"])
            rate = opening.get("elementary_rate_s-1")
            if probabilities and min(probabilities) >= 1. - 1e-12:
                disposition = "PHASE_CERTAIN_COMMITMENT_DENSITY"
            elif probabilities:
                disposition = "FINITE_TRANSIENT_COMMITMENT_DENSITY"
            elif opening["kinetic_validity"] == "NO_METASTABLE_ESCAPE_PATH":
                disposition = "STABLE_NO_ESCAPE_PATH"
            elif rate is not None and float(rate) * duration >= 1.:
                disposition = "HIGH_BARRIER_ACTION_EXCEEDS_ONE_PER_PHASE"
            else:
                disposition = opening["kinetic_validity"]
            output.append({
                "candidate_id": candidate_id,
                "reference_factor": factor,
                "Kmax_MPa_sqrt_m": factor * kref,
                "state": opening["state"],
                "kinetic_validity": opening["kinetic_validity"],
                "barrier_kBT": opening.get("barrier_kBT"),
                "adiabatic_relaxation_to_phase_ratio":
                    opening.get("adiabatic_relaxation_to_phase_ratio"),
                "elementary_rate_s-1": rate,
                "high_barrier_action_per_phase":
                    None if rate is None else float(rate) * duration,
                "grid_points": json.dumps(grids),
                "committed_probabilities": json.dumps(probabilities),
                "mean_commitment_times_s": json.dumps(means),
                "disposition": disposition,
            })
    write(args.out, output)
    print(json.dumps({
        "rows": len(output),
        "stable": sum(row["disposition"] == "STABLE_NO_ESCAPE_PATH" for row in output),
        "phase_certain": sum(row["disposition"] == "PHASE_CERTAIN_COMMITMENT_DENSITY"
                             for row in output),
        "high_barrier_phase_action": sum(
            row["disposition"] == "HIGH_BARRIER_ACTION_EXCEEDS_ONE_PER_PHASE"
            for row in output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
