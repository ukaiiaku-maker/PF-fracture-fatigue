#!/usr/bin/env python3
"""Bounded causal audit of the V5 child-tip radius ownership and rate law.

The interventions are software-causal peers at one accepted child state.  They
are not re-equilibrated physical loading cases and cannot qualify a new source
tensor.  Their purpose is to determine whether the accepted continuation law
actually consumes ``r_tip_m`` or silently substitutes the cavity radius.
"""
import argparse
from dataclasses import replace
import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture import voiding_production_v5 as production
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint
from arrhenius_fracture.closure_mechanics_evidence import canonical_data
from arrhenius_fracture.sharp_front import (
    FrontConfig, FrontEngine, default_cleavage_barrier, default_emission_barrier,
)
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
from arrhenius_fracture.voiding_v5 import replace_cavity


def source_location(function):
    lines, first = inspect.getsourcelines(function)
    return {"path": str(Path(inspect.getsourcefile(function)).relative_to(ROOT)),
            "function": function.__name__, "first_line": first,
            "last_line": first + len(lines) - 1}


def replace_child_radius(state, branch_id, factor):
    owned = {key: dict(value) for key, value in state.tip_process_state["by_branch"].items()}
    owned[branch_id]["r_tip_m"] *= factor
    tip_state = {**state.tip_process_state, "by_branch": owned}
    branches = tuple(
        replace(branch, local_state={**branch.local_state,
                                     "r_tip_m": owned[branch_id]["r_tip_m"]})
        if branch.branch_id == branch_id else branch
        for branch in state.crack_network.branches
    )
    return replace(state, tip_process_state=tip_state,
                   crack_network=replace(state.crack_network, branches=branches))


def replace_void_radius(state, factor):
    cavity = state.void_state.cavities[0]
    radius = cavity.radius_m * factor
    area = math.pi * radius**2
    delta = area - cavity.area_m2
    changed = replace(cavity, radius_m=radius, area_m2=area,
                      inventory_area_m2=cavity.inventory_area_m2 + delta)
    voids = replace_cavity(state.void_state, changed)
    voids = replace(voids,
        available_defect_inventory_area_m2=voids.available_defect_inventory_area_m2-delta,
        consumed_defect_inventory_area_m2=voids.consumed_defect_inventory_area_m2+delta)
    return replace(state, void_state=voids)


def rate_rows(state, tensor, temperature_K):
    engine = FrontEngine(FrontConfig(), default_cleavage_barrier(),
                         default_emission_barrier(state.material.b),
                         state.material.G, state.material.nu, state.material.b)
    actual = {row["candidate_id"]: row for row in
              production.directional_clock_rates(state, tensor, temperature_K=temperature_K)}
    rows = []
    for candidate in state.competition.candidates:
        normal = np.asarray(candidate.normal_xy, dtype=float)
        opening = max(float(normal @ tensor @ normal), 0.0)
        cleavage_rate, microscopic_raw, cleavage_barrier = engine.lambda_cleave(opening, temperature_K)
        emission_rate, emission_stress, emission_barrier = engine.lambda_emit(opening, temperature_K)
        measured = actual[candidate.candidate_id]
        rows.append({
            "candidate_id": candidate.candidate_id,
            "direction_xy": list(candidate.direction_xy),
            "normal_xy": list(candidate.normal_xy),
            "threshold_action": next(h.current_threshold_action for h in state.competition.hazard_states
                                      if h.candidate_id == candidate.candidate_id),
            "accumulated_action": next(h.action for h in state.competition.hazard_states
                                       if h.candidate_id == candidate.candidate_id),
            "resolved_opening_stress_Pa": opening,
            "cleavage_barrier_J": cleavage_barrier,
            "emission_barrier_J": emission_barrier,
            "microscopic_raw_cleavage_rate_s": microscopic_raw,
            "source_law_cleavage_rate_s": cleavage_rate,
            "effective_cleavage_rate_s": measured["effective_rate_s"],
            "emission_rate_s": emission_rate,
            "emission_effective_stress_Pa": emission_stress,
            "crossing_time_s": measured["crossing_time_s"],
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_pair", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--temperature-K", type=float, default=900.0)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite evidence")
    if subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip():
        raise RuntimeError("r_tip causal evidence requires a clean committed implementation")
    manifest_path = args.source_pair / "sha256_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    actual_manifest = {str(path.relative_to(args.source_pair)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in args.source_pair.rglob("*")
                       if path.is_file() and path.name != manifest_path.name}
    if actual_manifest != manifest:
        raise ValueError("source-pair manifest mismatch")

    state = restore_checkpoint(args.source_pair / "child_or_rejected_state.json")
    child_id = state.tip_process_state["active_branch_id"]
    child = state.crack_network.branch(child_id)
    tensor, element_ids = production.crack_tip_tensor(state, branch_id=child_id)
    tensor = np.asarray(tensor, dtype=float)
    baseline_rng = canonical_data(state.rng_state)
    baseline_void_radius = state.void_state.cavities[0].radius_m
    baseline_tip_radius = state.tip_process_state["by_branch"][child_id]["r_tip_m"]

    peers = []
    for factor in (0.75, 1.0, 1.25):
        peer = replace_child_radius(state, child_id, factor)
        rows = rate_rows(peer, tensor, args.temperature_K)
        operations = []
        after, event, operations, audit = production.downstream_front_transaction(
            peer, continuation=True, operation_log=operations)
        peers.append({
            "intervention": "child_r_tip_m", "factor": factor,
            "r_tip_before_event_m": peer.tip_process_state["by_branch"][child_id]["r_tip_m"],
            "R_void_m": peer.void_state.cavities[0].radius_m,
            "candidate_rates": rows,
            "selected_event_class": "physical_cleavage" if event is not None and event.accepted else None,
            "selected_candidate_id": audit.get("candidate_id"),
            "event_accepted": bool(event is not None and event.accepted),
            "renewed_r_tip_after_event_m": after.tip_process_state["by_branch"][child_id]["r_tip_m"],
            "terminal_fingerprint": complete_accepted_state_fingerprint(after),
            "operation_trace": operations,
            "fixed_inputs": {
                "source_tensor_Pa": tensor.tolist(), "source_element_ids": list(element_ids),
                "candidate_ids": [item.candidate_id for item in peer.competition.candidates],
                "thresholds": [item.current_threshold_action for item in peer.competition.hazard_states],
                "rng_state": baseline_rng, "temperature_K": args.temperature_K,
            },
        })

    reciprocal = replace_void_radius(state, 0.75)
    reciprocal_rows = rate_rows(reciprocal, tensor, args.temperature_K)
    baseline_rows = peers[1]["candidate_rates"]
    rate_keys = ("cleavage_barrier_J", "emission_barrier_J",
                 "microscopic_raw_cleavage_rate_s", "source_law_cleavage_rate_s",
                 "effective_cleavage_rate_s", "emission_rate_s", "crossing_time_s")
    r_tip_signatures = [[tuple(row[key] for key in rate_keys) for row in peer["candidate_rates"]]
                        for peer in peers]
    reciprocal_exact = all(tuple(row[key] for key in rate_keys) ==
                           tuple(base[key] for key in rate_keys)
                           for row, base in zip(reciprocal_rows, baseline_rows))
    r_tip_exact = r_tip_signatures[0] == r_tip_signatures[1] == r_tip_signatures[2]

    graph = {
        "schema": "v5.r-tip-consumer-graph/1",
        "nodes": [
            {"id": "tip_process_state.r_tip_m", "kind": "owned_state"},
            {"id": "child_branch.local_state.r_tip_m", "kind": "owned_state_mirror"},
            {"id": "child_source_tensor", "kind": "measured_FEM_source"},
            {"id": "directional_clock_rates", "kind": "kinetic_consumer",
             **source_location(production.directional_clock_rates)},
            {"id": "FrontEngine.lambda_cleave", "kind": "accepted_constitutive_law",
             **source_location(FrontEngine.lambda_cleave)},
            {"id": "downstream_front_transaction", "kind": "event_and_geometry_consumer",
             **source_location(production.downstream_front_transaction)},
        ],
        "edges": [
            {"from": "child_source_tensor", "to": "directional_clock_rates", "role": "stress_input"},
            {"from": "directional_clock_rates", "to": "FrontEngine.lambda_cleave", "role": "resolved_opening_and_temperature"},
            {"from": "directional_clock_rates", "to": "downstream_front_transaction", "role": "rate_and_crossing_time"},
            {"from": "tip_process_state.r_tip_m", "to": "child_branch.local_state.r_tip_m", "role": "ownership_mirror"},
            {"from": "tip_process_state.r_tip_m", "to": "downstream_front_transaction", "role": "preserved_state_only"},
        ],
        "absent_edges": [
            {"from": "tip_process_state.r_tip_m", "to": "directional_clock_rates"},
            {"from": "tip_process_state.r_tip_m", "to": "FrontEngine.lambda_cleave"},
            {"from": "R_void", "to": "directional_clock_rates"},
        ],
    }
    gates = {
        "controlled_peers_hold_required_inputs_fixed": all(
            peer["R_void_m"] == baseline_void_radius and
            peer["fixed_inputs"] == peers[1]["fixed_inputs"] for peer in peers),
        "all_controlled_events_complete": all(peer["event_accepted"] for peer in peers),
        "r_tip_changes_accepted_rate_or_barrier": not r_tip_exact,
        "reciprocal_R_void_not_substituted_for_r_tip": reciprocal_exact,
        "r_tip_distinct_from_R_void": baseline_tip_radius != baseline_void_radius,
    }
    classification = ("R_TIP_CAUSALLY_USED" if gates["r_tip_changes_accepted_rate_or_barrier"]
                      else "MISSING_ACCEPTED_R_TIP_CONSTITUTIVE_LINK")
    report = {
        "schema": "v5.r-tip-causal-use/1",
        "executed_code_sha": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT,
                                                     text=True).strip(),
        "source_pair_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "scope": "SOFTWARE_CAUSAL_INTERVENTIONS_AT_ONE_FIXED_ACCEPTED_CHILD_STATE",
        "child_branch_id": child_id, "child_tip_m": list(child.tip),
        "baseline_r_tip_m": baseline_tip_radius, "baseline_R_void_m": baseline_void_radius,
        "consumer_graph": graph, "r_tip_peers": peers,
        "reciprocal_R_void_peer": {
            "factor": 0.75, "r_tip_m": reciprocal.tip_process_state["by_branch"][child_id]["r_tip_m"],
            "R_void_m": reciprocal.void_state.cavities[0].radius_m,
            "candidate_rates": reciprocal_rows, "source_tensor_held_fixed": True,
            "source_state_held_fixed": True, "rates_and_barriers_exactly_equal": reciprocal_exact,
        },
        "gates": gates, "classification": classification,
        "passed": classification == "R_TIP_CAUSALLY_USED" and all(gates.values()),
    }
    args.output.mkdir(parents=True)
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(canonical_data(report), sort_keys=True, indent=2,
                                      allow_nan=False) + "\n")
    (args.output / "sha256_manifest.json").write_text(json.dumps({
        report_path.name: hashlib.sha256(report_path.read_bytes()).hexdigest()},
        sort_keys=True, indent=2) + "\n")
    print(json.dumps({"classification": classification, "gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()
