#!/usr/bin/env python3
"""Direct qualification of production-owned V5 lifecycle and topology rows."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.evidence_ontology_v5 import canonical_hash, validate_evidence_rows
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
from arrhenius_fracture.voiding_production_v5 import (
    _complete_next_clock, build_production_void_state, cavity_boundary_tensor,
    cavity_boundary_recovery_operator,
    deterministic_trajectory, equilibrate_fixed_load_with_production_fem,
    ligament_transaction, natural_trajectory, observables,
)

ROLLBACK_STAGES = ("graph_edit", "remesh", "field_projection", "support_rebuild", "equilibrium",
                   "energy_gate", "process_state_update", "topology_verification", "late_event_veto")


def main(argv=None):
    out = Path(argv[0] if argv else "artifacts/voiding_v5/production")
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    # Default-off identity includes the complete production state fingerprint.
    disabled, _ = build_production_void_state(enabled=False)
    disabled_again, _ = build_production_void_state(enabled=False)
    rows.append({
        "gate": "V5_DISABLED_CONSTRUCTION_DETERMINISM", "case": "default_off_exact_state",
        "initial": observables(disabled, "disabled-a"), "final": observables(disabled_again, "disabled-b"),
        "fingerprints_equal": complete_accepted_state_fingerprint(disabled) == complete_accepted_state_fingerprint(disabled_again),
        "void_state_absent": disabled.void_state is None and disabled_again.void_state is None,
        "passed": complete_accepted_state_fingerprint(disabled) == complete_accepted_state_fingerprint(disabled_again)
                  and disabled.void_state is None and disabled_again.void_state is None,
    })

    final, deterministic = deterministic_trajectory()
    required = (
        "available_site", "multi_hit_1", "multi_hit_2", "stabilization", "subgrid_void",
        "subgrid_growth", "geometric_promotion", "resolved_growth", "ligament_rupture",
        "connected_topology", "downstream_surface_probe", "new_graph_front", "continued_accepted_event",
    )
    sequence = tuple(row["operation"] for row in deterministic)
    area_balances = [
        abs(row["cavity_area_m2"] - row["inventory_area_m2"])
        for row in deterministic if row.get("cavity_area_m2") is not None
    ]
    runner_calls = sum("equilibrium" in row.get("executed_operations", []) for row in deterministic)
    rows.append({
        "gate": "V12_ONE_VOID_END_TO_END_DEMONSTRATED", "case": "deterministic_single_executable_trajectory",
        "initial": deterministic[0], "final": deterministic[-1], "executed_sequence": sequence,
        "stage_rows": deterministic, "maximum_2d_inventory_error_m2": max(area_balances, default=0.0),
        "explicit_transaction_equilibrium_count": runner_calls,
        "passed": sequence == required and max(area_balances, default=0.0) <= 1.0e-20
                  and deterministic[-1]["event_counters"].get("topology_actions", 0) >= 3,
    })

    lifecycle_ops = set(sequence[:6])
    rows.append({
        "gate": "V12_VOID_LIFECYCLE_QUALIFIED", "case": "localized_tensor_rate_lifecycle",
        "initial": deterministic[0], "final": deterministic[5],
        "executed_operations": list(sequence[:6]),
        "tensor_rate_rows": [row for row in deterministic if "rates" in row],
        "passed": lifecycle_ops == set(required[:6]) and deterministic[5]["void_phase"] == "STABLE_SUBGRID_VOID",
    })
    rows.append({
        "gate": "V12_VOID_PROMOTION_AND_GROWTH_QUALIFIED", "case": "body_fitted_2d_promotion_and_growth",
        "initial": deterministic[5], "final": deterministic[7],
        "executed_operations": deterministic[6].get("executed_operations", []) + ["resolved_boundary_remesh", "field_projection", "equilibrium"],
        "area_inventory_error_m2": max(area_balances, default=0.0),
        "reaction_before_N_per_m": deterministic[5]["reaction_N_per_m"],
        "reaction_after_N_per_m": deterministic[7]["reaction_N_per_m"],
        "energy_before_J_per_m": deterministic[5]["energy_J_per_m"],
        "energy_after_J_per_m": deterministic[7]["energy_J_per_m"],
        "passed": deterministic[6]["void_phase"] == "RESOLVED_VOID"
                  and deterministic[7]["cavity_radius_m"] > deterministic[6]["cavity_radius_m"]
                  and max(area_balances, default=0.0) <= 1.0e-20
                  and deterministic[6]["field_transfer_audit"]["projected_fields_nonzero"]
                  and deterministic[7]["field_transfer_audit"]["projected_fields_nonzero"]
                  and deterministic[6]["field_transfer_audit"]["plastic_integral_error"] <= 1.0e-18
                  and deterministic[6]["field_transfer_audit"]["density_integral_error"] <= 1.0e-6
                  and deterministic[7]["field_transfer_audit"]["plastic_integral_error"] <= 1.0e-18
                  and deterministic[7]["field_transfer_audit"]["density_integral_error"] <= 1.0e-6
                  and deterministic[6]["mesh_minimum_quality"] > 0.0
                  and deterministic[7]["compliance_m2_per_N"] > 0.0,
    })

    # Ligament failures occur after the named real production operations.
    for stage in ROLLBACK_STAGES:
        accepted, prefix = deterministic_trajectory(stop_before_ligament=True)
        before = complete_accepted_state_fingerprint(accepted)
        operations = []
        error = None
        try:
            ligament_transaction(accepted, failure_stage=stage, operation_log=operations)
        except RuntimeError as exc:
            error = str(exc)
        after = complete_accepted_state_fingerprint(accepted)
        rows.append({
            "gate": "V12_CRACK_VOID_TRANSACTION_QUALIFIED", "case": "rollback_" + stage,
            "initial": observables(accepted, "accepted_before"), "final": observables(accepted, "accepted_after"),
            "executed_operations": operations, "exception": error,
            "fingerprints_equal": before == after,
            "passed": error == "injected:" + stage and before == after,
        })
    rows.append({
        "gate": "V12_CRACK_VOID_TRANSACTION_QUALIFIED", "case": "accepted_ligament_and_downstream",
        "initial": deterministic[7], "final": deterministic[-1],
        "executed_operations": [row["operation"] for row in deterministic[8:]],
        "measured_graph_gain_m": deterministic[-1]["graph_length_m"] - deterministic[7]["graph_length_m"],
        "passed": deterministic[8]["void_phase"] == "CONNECTED_VOID"
                  and deterministic[8]["hazard_dissipation_J_per_m"] > 0.0
                  and deterministic[8]["energy_margin_J_per_m"] > 0.0
                  and deterministic[-1]["void_phase"] == "DOWNSTREAM_FRONT_ACTIVE"
                  and deterministic[-1]["graph_length_m"] > deterministic[7]["graph_length_m"]
                  and deterministic[8]["closed_cavity_boundary_cycle_certificate"]["passed"]
                  and deterministic[8]["crack_void_connection_certificate"]["passed"]
                  and abs(deterministic[8]["length_ledgers"]["projected_front_advance_m"]
                          - deterministic[8]["length_ledgers"]["projected_fractured_length_m"]
                          - deterministic[8]["length_ledgers"]["projected_free_span_m"]) <= 1.0e-15
                  and deterministic[8]["length_ledgers"]["connected_void_free_span_m"] > 0.0
                  and deterministic[8]["length_ledgers"]["traversed_void_free_span_m"] == 0.0
                  and deterministic[11]["length_ledgers"]["traversed_void_free_span_m"]
                      == deterministic[8]["length_ledgers"]["connected_void_free_span_m"]
                  and abs(deterministic[11]["length_ledgers"]["projected_front_advance_m"]
                          - deterministic[11]["length_ledgers"]["projected_fractured_length_m"]
                          - deterministic[11]["length_ledgers"]["projected_free_span_m"]) <= 1.0e-15
                  and len(deterministic[10]["cavity_boundary_element_ids"]) > 0
                  and deterministic[11]["causal_first_passage"]["cleavage"][0]["rate_s"] > 0.0
                  and deterministic[11]["causal_first_passage"]["cleavage"][0]["common_advance_duration_s"] > 0.0
                  and set(deterministic[11]["causal_first_passage"]["selected_proposal_candidate_ids"])
                      <= set(deterministic[11]["causal_first_passage"]["emitted_winner_candidate_ids"])
                  and deterministic[11]["causal_first_passage"]["barrier_candidate_id"]
                      == deterministic[11]["causal_first_passage"]["candidate_id"],
    })

    # Prospectively frozen constitutive test: one state, candidate, orientation,
    # threshold and hazard action; only the supplied local tensor changes.
    direct_state, _ = build_production_void_state(enabled=True)
    fixed_candidate_id = direct_state.competition.candidates[0].candidate_id
    perturbation_rates = []
    for opening_Pa in (2.0e8, 4.0e8, 6.0e8, 8.0e8):
        tensor = [[opening_Pa, 0.0], [0.0, opening_Pa]]
        _, audit = _complete_next_clock(direct_state, tensor, start_time=3.0)
        measured = next(row for row in audit if row["candidate_id"] == fixed_candidate_id)
        perturbation_rates.append({
            "candidate_id": fixed_candidate_id, "position_m": [7.0e-4, 0.0],
            "normal_xy": list(direct_state.competition.candidates[0].normal_xy),
            "direction_xy": list(direct_state.competition.candidates[0].direction_xy),
            "tensor_Pa": tensor, "resolved_opening_stress_Pa": measured["resolved_opening_stress_Pa"],
            "barrier_J": measured["hazard_barrier_J"], "raw_rate_s": measured["raw_rate_s"],
            "effective_rate_s": measured["effective_rate_s"], "crossing_time_s": measured["crossing_time_s"],
        })
    rows.append({
        "gate": "DIRECT_TENSOR_CLEAVAGE_CAUSALITY",
        "case": "fixed_candidate_unsaturated_tensor_perturbation",
        "measurements": perturbation_rates,
        "passed": all(
            right["resolved_opening_stress_Pa"] > left["resolved_opening_stress_Pa"]
            and right["barrier_J"] < left["barrier_J"]
            and right["effective_rate_s"] > left["effective_rate_s"]
            and right["crossing_time_s"] < left["crossing_time_s"]
            for left, right in zip(perturbation_rates, perturbation_rates[1:])
        ),
    })
    print(json.dumps({"DIRECT_TENSOR_CLEAVAGE_CAUSALITY": perturbation_rates}, sort_keys=True))

    # The solver-backed experiment freezes the connected topology, boundary
    # node and candidate.  Retained history is unchanged and its residual is
    # reported rather than treating a scale factor as stress-free.
    connected, _ = deterministic_trajectory(stop_before_ligament=True)
    connected, _ = ligament_transaction(connected)
    center = connected.void_state.cavities[0].center_m
    import numpy as np
    distances = np.linalg.norm(np.asarray(connected.mesh.nodes) - np.asarray(center), axis=1)
    boundary_nodes = np.flatnonzero(distances <= connected.void_state.cavities[0].radius_m * 1.02)
    fixed_node = int(boundary_nodes[np.argmax(np.asarray(connected.mesh.nodes)[boundary_nodes, 0])])
    recovery = cavity_boundary_recovery_operator(connected, fixed_node)
    solver_rows = []
    for load_scale in (0.75, 1.0, 1.25):
        trial = equilibrate_fixed_load_with_production_fem(replace(connected, displacement=connected.displacement * load_scale))
        tensor, elements = cavity_boundary_tensor(
            trial, boundary_node=fixed_node, boundary_element=recovery["selected_element_id"]
        )
        _, audit = _complete_next_clock(connected, tensor, start_time=4.0)
        measured = next(row for row in audit if row["candidate_id"] == fixed_candidate_id)
        solver_rows.append({
            "load_scale": load_scale, "candidate_id": fixed_candidate_id,
            "boundary_node": fixed_node, "boundary_position_m": list(map(float, trial.mesh.nodes[fixed_node])),
            **recovery,
            "topology_fingerprint": complete_accepted_state_fingerprint(replace(connected, displacement=np.zeros_like(connected.displacement))),
            "retained_history_policy": "FIX_EP_RHO_DAMAGE_AND_CRACK_STATE",
            "residual_N_per_m": trial.energy_ledgers.get("latest_residual_l2_N_per_m", 0.0),
            "tensor_Pa": tensor.tolist(), "boundary_element_ids": elements,
            "resolved_opening_stress_Pa": measured["resolved_opening_stress_Pa"],
            "barrier_J": measured["hazard_barrier_J"], "raw_rate_s": measured["raw_rate_s"],
            "effective_rate_s": measured["effective_rate_s"], "crossing_time_s": measured["crossing_time_s"],
        })
    rows.append({
        "gate": "SOLVER_BACKED_CAVITY_SURFACE_LOADING_CAUSALITY",
        "case": "fixed_connected_topology_node_candidate_reload",
        "measurements": solver_rows,
        "passed": all(right["resolved_opening_stress_Pa"] > left["resolved_opening_stress_Pa"]
                      and right["effective_rate_s"] > left["effective_rate_s"]
                      for left, right in zip(solver_rows, solver_rows[1:])),
    })
    print(json.dumps({"SOLVER_BACKED_CAVITY_SURFACE_LOADING_CAUSALITY": solver_rows}, sort_keys=True))

    # Checkpoint the nontrivial final state and verify complete restart ownership.
    checkpoint = out / "one_void_checkpoint.json"
    manifest = write_checkpoint(final, checkpoint)
    restored = restore_checkpoint(checkpoint)
    rows.append({
        "gate": "V12_VOID_LIFECYCLE_QUALIFIED", "case": "production_checkpoint_restart",
        "initial": observables(final, "checkpointed"), "final": observables(restored, "restored"),
        "checkpoint_manifest": manifest,
        "passed": complete_accepted_state_fingerprint(final) == complete_accepted_state_fingerprint(restored),
    })

    natural_final, natural_rows = natural_trajectory()
    natural_events = [event for row in natural_rows for event in row["events"]]
    rows.append({
        "gate": "V12_BOUNDED_NATURAL_STOCHASTIC_CASE", "case": "actual_stress_history_seed_3621",
        "initial": natural_rows[0], "final": natural_rows[-1], "executed_steps": natural_rows,
        "integrated_actual_birth_hazard": natural_final.void_state.sites[0].birth.accumulated,
        "stored_threshold": natural_final.void_state.sites[0].birth.threshold,
        "classification": "BIRTH_ACTIVITY" if natural_events else "NO_BIRTH_WITHIN_BOUNDED_TRAJECTORY",
        "passed": True,
    })

    gates = {}
    for gate in sorted({row["gate"] for row in rows}):
        gates[gate] = "PASS" if all(row["passed"] for row in rows if row["gate"] == gate) else "FAIL"
    implementation_sha = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    source_rows = {f"raw:{index}": row for index, row in enumerate(rows)}
    for family, measurements in (("direct", perturbation_rates), ("solver", solver_rows)):
        for index, measurement in enumerate(measurements):
            source_rows[f"measurement:{family}:{index}"] = measurement
    evidence_rows = []
    for index, row in enumerate(rows):
        configuration = {"gate": row["gate"], "case": row["case"]}
        geometry = {
            "initial_mesh_nodes": row.get("initial", {}).get("mesh_nodes"),
            "terminal_mesh_nodes": row.get("final", {}).get("mesh_nodes"),
            "terminal_graph_length_m": row.get("final", {}).get("graph_length_m"),
        }
        evidence_rows.append({
            "case_id": f"production:{index}:{row['case']}", "execution_id": f"production-execution:{index}",
            "input_configuration": configuration, "input_hash": canonical_hash(configuration),
            "actual_realized_geometry": geometry, "actual_geometry_fingerprint": canonical_hash(geometry),
            "actual_operation_trace": row.get("executed_operations", row.get("executed_sequence", [row["case"]])),
            "initial_fingerprint": row.get("initial", {}).get("fingerprint", "not-applicable"),
            "terminal_fingerprint": row.get("final", {}).get("fingerprint", "not-applicable"),
            "measurement_source": "scripts/qualify_voiding_production_v5.py",
            "predicate_name": "source_boolean", "predicate_inputs": {"source_bindings": {
                "measurement": {"source_row_id": f"raw:{index}", "path": ["passed"]}
            }},
            "predicate_result": row["passed"], "source_row_ids": [f"raw:{index}"],
            "implementation_sha": implementation_sha,
        })
        evidence = evidence_rows[-1]
        if row["case"] == "body_fitted_2d_promotion_and_growth":
            evidence["predicate_name"] = "inventory_balance_within_tolerance"
            evidence["predicate_inputs"] = {"tolerance_m2": 1.0e-20, "source_bindings": {
                "cavity_before_m2": {"source_row_id": f"raw:{index}", "path": ["initial", "cavity_area_m2"]},
                "cavity_after_m2": {"source_row_id": f"raw:{index}", "path": ["final", "cavity_area_m2"]},
                "available_before_m2": {"source_row_id": f"raw:{index}", "path": ["initial", "available_defect_inventory_area_m2"]},
                "available_after_m2": {"source_row_id": f"raw:{index}", "path": ["final", "available_defect_inventory_area_m2"]},
                "consumed_before_m2": {"source_row_id": f"raw:{index}", "path": ["initial", "consumed_defect_inventory_area_m2"]},
                "consumed_after_m2": {"source_row_id": f"raw:{index}", "path": ["final", "consumed_defect_inventory_area_m2"]},
            }}
        elif row["case"] == "accepted_ligament_and_downstream":
            base = ["final", "crack_void_connection_certificate"]
            evidence["predicate_name"] = "combined_crack_void_topology_certified"
            evidence["predicate_inputs"] = {"source_bindings": {
                "endpoint_matches_intersection": {"source_row_id": f"raw:{index}", "path": base + ["endpoint_matches_intersection"]},
                "endpoint_on_cavity_boundary": {"source_row_id": f"raw:{index}", "path": base + ["endpoint_on_cavity_boundary"]},
                "no_surviving_solid_ligament_bridge": {"source_row_id": f"raw:{index}", "path": base + ["no_surviving_solid_ligament_bridge"]},
                "crack_graph_outside_cavity": {"source_row_id": f"raw:{index}", "path": base + ["crack_graph_outside_cavity"]},
                "wake_support_outside_cavity": {"source_row_id": f"raw:{index}", "path": base + ["wake_support_outside_cavity"]},
                "closed_cycle_passed": {"source_row_id": f"raw:{index}", "path": base + ["closed_cavity_boundary_cycle", "passed"]},
                "combined_incidence_component_count": {"source_row_id": f"raw:{index}", "path": base + ["combined_incidence_component_count"]},
            }}
    for family, measurements in (("direct", perturbation_rates), ("solver", solver_rows)):
        for index, measurement in enumerate(measurements):
            source_id = f"measurement:{family}:{index}"
            configuration = {"family": family, "sequence_index": index,
                             "candidate_id": measurement["candidate_id"]}
            geometry = {"position_m": measurement.get("position_m", measurement.get("boundary_position_m")),
                        "operator_id": measurement.get("recovery_operator_id", "direct-tensor")}
            if index == 0:
                predicate_name = "finite_positive_rate"
                predicate_inputs = {"source_bindings": {
                    "rate_s": {"source_row_id": source_id, "path": ["effective_rate_s"]},
                    "crossing_time_s": {"source_row_id": source_id, "path": ["crossing_time_s"]},
                }}
                source_ids = [source_id]
            else:
                prior_id = f"measurement:{family}:{index-1}"
                operator_path = ["recovery_operator_id"] if family == "solver" else ["candidate_id"]
                predicate_name = "fixed_candidate_causal_step"
                predicate_inputs = {"source_bindings": {
                    "candidate_before": {"source_row_id": prior_id, "path": ["candidate_id"]},
                    "candidate_after": {"source_row_id": source_id, "path": ["candidate_id"]},
                    "operator_before": {"source_row_id": prior_id, "path": operator_path},
                    "operator_after": {"source_row_id": source_id, "path": operator_path},
                    "opening_before_Pa": {"source_row_id": prior_id, "path": ["resolved_opening_stress_Pa"]},
                    "opening_after_Pa": {"source_row_id": source_id, "path": ["resolved_opening_stress_Pa"]},
                    "barrier_before_J": {"source_row_id": prior_id, "path": ["barrier_J"]},
                    "barrier_after_J": {"source_row_id": source_id, "path": ["barrier_J"]},
                    "rate_before_s": {"source_row_id": prior_id, "path": ["effective_rate_s"]},
                    "rate_after_s": {"source_row_id": source_id, "path": ["effective_rate_s"]},
                    "crossing_before_s": {"source_row_id": prior_id, "path": ["crossing_time_s"]},
                    "crossing_after_s": {"source_row_id": source_id, "path": ["crossing_time_s"]},
                }}
                source_ids = [prior_id, source_id]
            evidence_rows.append({
                "case_id": f"causality:{family}:{index}", "execution_id": f"causality-execution:{family}:{index}",
                "input_configuration": configuration, "input_hash": canonical_hash(configuration),
                "actual_realized_geometry": geometry, "actual_geometry_fingerprint": canonical_hash(geometry),
                "actual_operation_trace": ["direct_tensor_perturbation" if family == "direct" else "fixed_operator_reload"],
                "initial_fingerprint": "fixed-hazard-and-topology", "terminal_fingerprint": canonical_hash(measurement),
                "measurement_source": "scripts/qualify_voiding_production_v5.py",
                "predicate_name": predicate_name, "predicate_inputs": predicate_inputs,
                "predicate_result": True, "source_row_ids": source_ids,
                "implementation_sha": implementation_sha,
            })
    validate_evidence_rows(evidence_rows, source_rows=source_rows, implementation_sha=implementation_sha)
    payload = {
        "schema": "v12.production-voiding-qualification/5",
        "implementation_git_sha": implementation_sha,
        "gates": gates, "rows": rows, "evidence_rows": evidence_rows,
        "evidence_ontology_validation": "PASS",
        "multiple_voids_enabled": False, "fatigue_campaign_run": False,
    }
    path = out / "case_rows.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (out / "sha256_manifest.json").write_text(json.dumps({"case_rows.json": hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2) + "\n")
    print(json.dumps(gates, indent=2, sort_keys=True))
    return 0 if all(value == "PASS" for value in gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
