#!/usr/bin/env python3
"""Broad current-law cyclic-state map for autonomous R10 acquisition."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.cohesive_arrhenius import COHESIVE_KINETIC_CZM
from arrhenius_fracture.current_coupled_execution_r4 import R4_KERNEL_FAMILY
from arrhenius_fracture.czm_high_cycle_adapter_r10 import (
    CZMPhaseController, install_czm_owner, opening_snapshot,
)
from arrhenius_fracture.fatigue_v1 import FatigueWaveform
from arrhenius_fracture.opening_alternatives_r10 import OpeningAlternative
from arrhenius_fracture.persistent_site_high_cycle_state_v10230 import serialize_active_state
from arrhenius_fracture.persistent_site_poincare_v10230 import one_cycle_map
from arrhenius_fracture.retained_material_cohesive_r10 import (
    build_cohesive_current_engine, load_cohesive_binding,
)
from scripts.run_r10_coupled_opening_search import reference_load
from scripts.run_r10_opening_alternatives import mechanics, thermal_from


def event_reachable(snapshot: dict, normalized_clock_per_cycle: float) -> bool:
    """Use deterministic completion as well as the mixed survival clock.

    Athermal arms have no hazard clock.  Their peak-load connected branch can
    nevertheless already be mechanically complete, which the exact HCF
    consumer commits on the corresponding phase.  Treating only a clock >= 1
    as reachable incorrectly labels these arms quiet.
    """
    return (snapshot.get("branch_state") == "SEPARATION_COMPLETED" or
            normalized_clock_per_cycle >= 1.)


def write_csv(path: Path, rows: list[dict]) -> None:
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


def task(payload: dict) -> dict:
    row = payload["row"]
    temperature = payload["temperature_K"]
    phases = payload["phases"]
    cohesive = json.loads(row["cohesive_parameter_vector"])
    thermal = thermal_from(cohesive)
    _, kenv = mechanics(thermal, temperature)
    kref, _ = reference_load(row, temperature, phases)
    kmax = payload["factor"] * kref
    alternative = OpeningAlternative(
        row["selected_mode"], row["selected_shape"],
        "quasistatic_mechanical" if row["selected_mode"] == "cohesive_athermal"
        else "time_dependent_first_passage",
        None if row["selected_mode"] == "cohesive_athermal" else float(cohesive["activation_area_m2"]),
        None if row["selected_mode"] == "cohesive_athermal" else float(cohesive["attempt_frequency_s"]),
    )
    engine = build_cohesive_current_engine(
        load_cohesive_binding(Path(payload["registry"]), row["candidate_id"],
                              execution_identity=COHESIVE_KINETIC_CZM),
        kernel_family_path=R4_KERNEL_FAMILY,
    )
    install_czm_owner(engine, alternative, thermal, kenv, phases, payload["seed"],
                      frequency_Hz=payload["frequency_Hz"])
    wave = FatigueWaveform(kmax * 1e6, payload["R"], payload["frequency_Hz"])
    controller = CZMPhaseController(phases)
    initial_state = serialize_active_state(engine, waveform=wave, temperature_K=temperature).vector
    initial = opening_snapshot(engine, wave.Kmax, temperature)
    first = one_cycle_map(engine, controller, wave, temperature)
    successor = one_cycle_map(first.engine, controller, wave, temperature)
    end_state = successor.state_end.vector
    scale = np.maximum(np.maximum(np.abs(initial_state), np.abs(end_state)), 1.)
    rel = float(np.max(np.abs(end_state - initial_state) / scale))
    drift = float(np.linalg.norm(successor.state_end.vector - first.state_end.vector) /
                  max(np.linalg.norm(first.state_end.vector), 1.))
    final = opening_snapshot(successor.engine, wave.Kmax, temperature)
    action = float(successor.hazard_action_per_cycle)
    return {
        "candidate_id": row["candidate_id"],
        "source_material_candidate_id": row["source_material_candidate_id"],
        "complete_row_sha256": row["complete_row_sha256"],
        "plasticity_physics_sha256": row["plasticity_physics_sha256"],
        "cohesive_parameter_sha256": row["cohesive_parameter_sha256"],
        "search_design": row["search_design"], "shape": row["selected_shape"],
        "mode": row["selected_mode"], "mechanism_prediction": row["mechanism_prediction"],
        "thermal_prediction": row["thermal_prediction"],
        "temperature_K": temperature, "R": payload["R"],
        "frequency_Hz": payload["frequency_Hz"], "reference_factor": payload["factor"],
        "Kmax_MPa_sqrt_m": kmax, "DeltaK_MPa_sqrt_m": (1 - payload["R"]) * kmax,
        "initial_branch": initial["branch_state"], "final_branch": final["branch_state"],
        "initial_validity": initial["kinetic_validity"], "final_validity": final["kinetic_validity"],
        "initial_barrier_kBT": initial["barrier_kBT"], "final_barrier_kBT": final["barrier_kBT"],
        "action_per_cycle_second_cycle": action,
        "action_timescale_cycles": None if action <= 0 else 1 / action,
        "successor_state_drift": drift,
        "successor_state_timescale_cycles": None if drift <= 0 else 1 / drift,
        "two_cycle_componentwise_relative_change": rel,
        "mobile_count_initial": initial["mobile_count"], "mobile_count_final": final["mobile_count"],
        "tip_radius_m_initial": initial["tip_radius_m"], "tip_radius_m_final": final["tip_radius_m"],
        "separation_m_initial": initial["cohesive_separation_m"],
        "separation_m_final": final["cohesive_separation_m"],
        "one_cycle_event_reachable": event_reachable(
            initial, float(first.normalized_clock_per_cycle)),
        "second_cycle_event_reachable": event_reachable(
            final, float(successor.normalized_clock_per_cycle)),
        "unsupported": bool(successor.engine._r10_czm_unsupported),
        "unsupported_validities": json.dumps(sorted(successor.engine._r10_czm_unsupported_validities)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--factors", default=".84,.92,.97,1.01,1.06")
    parser.add_argument("--temperature-K", type=float, default=300.)
    parser.add_argument("--R", type=float, default=.1)
    parser.add_argument("--frequency-Hz", type=float, default=1000.)
    parser.add_argument("--phases", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026101801)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--mode", choices=("cohesive_athermal", "cohesive_mixed"),
                        help="Optionally reevaluate only one opening owner.")
    parser.add_argument("--candidate-ids",
                        help="Optional comma-separated execution-identity subset.")
    args = parser.parse_args(argv)
    with args.registry.open(newline="") as stream:
        registry = list(csv.DictReader(stream))
    if args.mode:
        registry = [row for row in registry if row["selected_mode"] == args.mode]
    if args.candidate_ids:
        requested = {value.strip() for value in args.candidate_ids.split(",") if value.strip()}
        registry = [row for row in registry if row["candidate_id"] in requested]
    if args.limit:
        registry = registry[:args.limit]
    factors = [float(x) for x in args.factors.split(",")]
    work = []
    for i, row in enumerate(registry):
        for j, factor in enumerate(factors):
            work.append({"registry": str(args.registry), "row": row, "factor": factor,
                         "temperature_K": args.temperature_K, "R": args.R,
                         "frequency_Hz": args.frequency_Hz, "phases": args.phases,
                         "seed": args.seed + 100 * i + j})
    output = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(task, item) for item in work]
        for index, future in enumerate(as_completed(futures), 1):
            output.append(future.result())
            if index % 50 == 0:
                print(json.dumps({"completed": index, "total": len(work)}), flush=True)
    output.sort(key=lambda r: (r["candidate_id"], r["reference_factor"]))
    write_csv(args.out, output)
    print(json.dumps({"rows": len(output),
                      "physical_vectors": len({r['source_material_candidate_id'] for r in output}),
                      "unsupported": sum(r["unsupported"] for r in output),
                      "eventful": sum(r["one_cycle_event_reachable"] or
                                      r["second_cycle_event_reachable"] for r in output)},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
