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
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.evidence_ontology_v5 import canonical_hash, validate_evidence_rows
from arrhenius_fracture.directional_competition_v11 import (
    CleavageCandidate, DirectionalCompetitionState, DirectionalHazardState,
)
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
from arrhenius_fracture.voiding_production_v5 import (
    _complete_next_clock, build_production_void_state, cavity_boundary_tensor,
    cavity_boundary_recovery_operator,
    crack_tip_tensor, deterministic_trajectory, downstream_front_transaction,
    equilibrate_fixed_load_with_production_fem, ligament_transaction,
    natural_trajectory, observables,
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
        "gate": "V5_CONTROLLED_REFERENCE_TRAJECTORY_SCREEN", "case": "deterministic_single_executable_trajectory",
        "initial": deterministic[0], "final": deterministic[-1], "executed_sequence": sequence,
        "stage_rows": deterministic, "maximum_2d_inventory_error_m2": max(area_balances, default=0.0),
        "explicit_transaction_equilibrium_count": runner_calls,
        "passed": sequence == required and max(area_balances, default=0.0) <= 1.0e-20
                  and deterministic[-1]["event_counters"].get("topology_actions", 0) >= 3,
    })

    lifecycle_ops = set(sequence[:6])
    rows.append({
        "gate": "V5_REFERENCE_LIFECYCLE_SCREEN", "case": "localized_tensor_rate_lifecycle",
        "initial": deterministic[0], "final": deterministic[5],
        "executed_operations": list(sequence[:6]),
        "tensor_rate_rows": [row for row in deterministic if "rates" in row],
        "passed": lifecycle_ops == set(required[:6]) and deterministic[5]["void_phase"] == "STABLE_SUBGRID_VOID",
    })
    rows.append({
        "gate": "V5_REFERENCE_PROMOTION_GROWTH_SCREEN", "case": "body_fitted_2d_promotion_and_growth",
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
            "gate": "V5_REFERENCE_TRANSACTION_SCREEN", "case": "rollback_" + stage,
            "initial": observables(accepted, "accepted_before"), "final": observables(accepted, "accepted_after"),
            "executed_operations": operations, "exception": error,
            "fingerprints_equal": before == after,
            "passed": error == "injected:" + stage and before == after,
        })
    rows.append({
        "gate": "V5_REFERENCE_TRANSACTION_SCREEN", "case": "accepted_ligament_and_downstream",
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
    rows.append({
        "gate": "V12_SINGLE_ACTIVE_FRONT_OWNERSHIP",
        "case": "connected_dormancy_then_downstream_sole_activation",
        "initial": deterministic[8], "final": deterministic[11],
        "connected_graph_active_ids": deterministic[8]["active_crack_branch_ids"],
        "connected_support_active_ids": deterministic[8]["support_active_tip_ids"],
        "downstream_graph_active_ids": deterministic[11]["active_crack_branch_ids"],
        "downstream_support_active_ids": deterministic[11]["support_active_tip_ids"],
        "executed_operations": ["root_status_change", "dormant_support_rebuild",
                                "downstream_child_activation", "support_rebuild"],
        "passed": deterministic[8]["active_crack_branch_ids"] == []
                  and deterministic[8]["support_active_tip_ids"] == []
                  and deterministic[11]["active_crack_branch_ids"] == ["void-front-1"]
                  and deterministic[11]["support_active_tip_ids"] == ["void-front-1"],
    })
    final_ledger = deterministic[-1]["length_ledgers"]
    rows.append({
        "gate": "V5_GENERALIZED_LENGTH_ACCOUNTING", "case": "complete_reference_length_identities",
        "initial": deterministic[7], "final": deterministic[-1],
        "physical_front_travel_m": final_ledger["physical_active_front_travel_m"],
        "physical_fractured_m": (final_ledger["fractured_ligament_length_m"]
                                  + final_ledger["ordinary_crack_fractured_length_m"]),
        "traversed_physical_void_m": final_ledger["traversed_void_free_span_m"],
        "projected_front_advance_m": final_ledger["projected_front_advance_m"],
        "projected_fractured_m": final_ledger["projected_fractured_length_m"],
        "projected_void_m": final_ledger["projected_free_span_m"],
        "tolerance_m": 1.0e-15,
        "passed": abs(final_ledger["physical_active_front_travel_m"]
                      - final_ledger["fractured_ligament_length_m"]
                      - final_ledger["ordinary_crack_fractured_length_m"]
                      - final_ledger["traversed_void_free_span_m"]) <= 1.0e-15
                  and abs(final_ledger["projected_front_advance_m"]
                          - final_ledger["projected_fractured_length_m"]
                          - final_ledger["projected_free_span_m"]) <= 1.0e-15,
    })
    rows.append({
        "gate": "V5_CHILD_SHARP_FRONT_HANDOFF", "case": "cavity_to_child_tip_causal_handoff",
        "initial": deterministic[10], "final": deterministic[-1],
        "first_source_kind": deterministic[11]["causal_first_passage"]["source_kind"],
        "continued_source_kind": deterministic[12]["causal_first_passage"]["source_kind"],
        "continued_source_front_id": deterministic[12]["causal_first_passage"]["source_front_id"],
        "active_branch_id": final.tip_process_state["active_branch_id"],
        "r_tip_m": final.tip_process_state["by_branch"]["void-front-1"]["r_tip_m"],
        "topology_stage": final.junction_process_state["latest_topology_certificate_stage"],
        "executed_operations": ["cavity_surface_first_passage", "fresh_tip_renewal",
                                "child_tip_first_passage", "topology_recertification"],
        "passed": deterministic[11]["causal_first_passage"]["source_kind"] == "cavity_surface"
                  and deterministic[12]["causal_first_passage"]["source_kind"] == "sharp_front"
                  and deterministic[12]["causal_first_passage"]["source_front_id"] == "void-front-1"
                  and final.tip_process_state["by_branch"]["void-front-1"]["r_tip_m"] > 0.0
                  and final.junction_process_state["latest_topology_certificate_stage"] == "POST_CONTINUATION",
    })
    for case, stage_index, expected_components in (
        ("topology_A_root_cavity_connection", 8, {"branch:b00000000", "cavity:void:site-1"}),
        ("topology_B_root_cavity_downstream_child", 11,
         {"branch:b00000000", "branch:void-front-1", "cavity:void:site-1"}),
        ("topology_C_post_child_continuation", 12,
         {"branch:b00000000", "branch:void-front-1", "cavity:void:site-1"}),
    ):
        certificate = deterministic[stage_index]["crack_void_connection_certificate"]
        rows.append({
            "gate": "V5_EVENTWISE_EXACT_TOPOLOGY", "case": case,
            "initial": deterministic[max(stage_index - 1, 0)], "final": deterministic[stage_index],
            "certificate": certificate,
            "expected_component_members": sorted(expected_components),
            "passed": certificate["passed"]
                      and set(certificate["combined_components"][0]) == expected_components
                      and certificate["exact_no_intact_node_or_element_path"]
                      and certificate["exact_triangle_polygon_interior_overlap_absent"],
        })
    rows.append({
        "gate": "V5_FREE_DOF_EQUILIBRIUM", "case": "terminal_equilibrium_decomposition",
        "initial": deterministic[0], "final": deterministic[-1],
        "free_residual_l2": deterministic[-1]["free_dof_residual_l2_N_per_m"],
        "constrained_reaction_l2": deterministic[-1]["constrained_reaction_l2_N_per_m"],
        "reaction_balance": deterministic[-1]["top_bottom_reaction_balance"],
        "energy_reaction_identity": deterministic[-1]["energy_reaction_identity"],
        "free_residual_relative_tolerance": 1.0e-8,
        "reaction_balance_tolerance": 3.0e-2,
        "energy_identity_tolerance": 1.0e-2,
        "passed": deterministic[-1]["free_dof_residual_l2_N_per_m"]
                      <= 1.0e-8 * deterministic[-1]["constrained_reaction_l2_N_per_m"]
                  and deterministic[-1]["top_bottom_reaction_balance"] <= 3.0e-2
                  and deterministic[-1]["energy_reaction_identity"] <= 1.0e-2,
    })

    fixed_path = ((0.0, 0.0), (0.0005725993004046688, 0.0))
    offset_results = []
    for offset in (1.0e-5, -1.0e-5):
        offset_state, offset_rows = deterministic_trajectory(
            stop_before_ligament=True, cavity_center_m=(7.0e-4, offset),
            crack_path_m=fixed_path,
        )
        offset_connected, _ = ligament_transaction(offset_state)
        offset_results.append((offset, offset_rows[-1], offset_connected))
        rows.append({
            "gate": "V5_TRUE_FIXED_CRACK_OFFSET_EVIDENCE",
            "case": "positive_offset" if offset > 0.0 else "negative_offset",
            "initial": offset_rows[-1], "final": observables(offset_connected, "connected_offset"),
            "offset_m": offset,
            "intersection_m": list(offset_connected.void_state.cavities[0].connection_entry_m),
            "connection_exit_m": list(offset_connected.void_state.cavities[0].connection_exit_m),
            "certified": offset_connected.junction_process_state["latest_crack_void_connection_certificate"]["passed"],
            "passed": offset_connected.junction_process_state["latest_crack_void_connection_certificate"]["passed"],
        })
    positive, negative = offset_results
    reaction_error = abs(positive[1]["reaction_N_per_m"] / negative[1]["reaction_N_per_m"] - 1.0)
    compliance_error = abs(positive[1]["compliance_m2_per_N"] / negative[1]["compliance_m2_per_N"] - 1.0)
    pos_entry = positive[2].void_state.cavities[0].connection_entry_m
    neg_entry = negative[2].void_state.cavities[0].connection_entry_m
    mirror_error = math.hypot(pos_entry[0] - neg_entry[0], pos_entry[1] + neg_entry[1])
    rows.append({
        "gate": "V5_TRUE_FIXED_CRACK_OFFSET_EVIDENCE", "case": "offset_pair_symmetry",
        "initial": positive[1], "final": negative[1],
        "positive_certified": positive[2].junction_process_state["latest_crack_void_connection_certificate"]["passed"],
        "negative_certified": negative[2].junction_process_state["latest_crack_void_connection_certificate"]["passed"],
        "reaction_relative_error": reaction_error, "reaction_tolerance": 5.0e-3,
        "compliance_relative_error": compliance_error, "compliance_tolerance": 5.0e-3,
        "mirrored_intersection_error_m": mirror_error, "geometry_tolerance_m": 1.0e-12,
        "passed": reaction_error <= 5.0e-3 and compliance_error <= 5.0e-3
                  and mirror_error <= 1.0e-12,
    })

    negative_connected = negative[2]
    dormant_source = negative_connected.junction_process_state["active_event_source"]
    zero_partition_rows = []
    for partitions in (1, 2, 4, 8, 16):
        trial = negative_connected
        initial_competition = trial.competition
        initial_rng = trial.rng_state
        initial_graph = trial.crack_network
        initial_support = trial.v12_support_state
        for _ in range(partitions):
            trial, audit = _complete_next_clock(
                trial, np.zeros((2, 2)), source_kind="cavity_surface",
                source_cavity_id=trial.void_state.cavities[0].cavity_id,
                source_boundary_site_id="connection_exit",
                source_position_m=trial.void_state.cavities[0].connection_exit_m,
                source_probe_identity={"kind": "direct_cavity_boundary_tensor"},
                maximum_advance_duration_s=16.0 / partitions,
            )
        zero_partition_rows.append({
            "partitions": partitions,
            "all_rates_zero": all(row["effective_rate_s"] == 0.0 for row in audit),
            "competition_unchanged": trial.competition == initial_competition,
            "rng_unchanged": trial.rng_state == initial_rng,
            "graph_unchanged": trial.crack_network == initial_graph,
            "support_unchanged": trial.v12_support_state == initial_support,
        })
    rows.append({
        "gate": "V5_ZERO_DRIVE_CONNECTED_POLICY",
        "case": "negative_offset_connected_zero_drive_partitions",
        "initial": negative[1], "final": observables(negative_connected, "connected_zero_drive"),
        "candidate_source_states": list(dormant_source["candidate_source_states"]),
        "active_graph_tip_ids": list(negative_connected.crack_network.active_tip_ids),
        "active_support_tip_ids": list(negative_connected.v12_support_state.active_tip_identities),
        "child_branch_exists": any(branch.branch_id == "void-front-1"
                                   for branch in negative_connected.crack_network.branches),
        "partition_measurements": zero_partition_rows,
        "classification": "CONNECTED_VOID_ZERO_DOWNSTREAM_DRIVE",
        "passed": all(
            row["all_rates_zero"] and row["competition_unchanged"] and row["rng_unchanged"]
            and row["graph_unchanged"] and row["support_unchanged"]
            for row in zero_partition_rows
        ) and not negative_connected.crack_network.active_tip_ids
          and not negative_connected.v12_support_state.active_tip_identities
          and not any(branch.branch_id == "void-front-1"
                      for branch in negative_connected.crack_network.branches)
          and all(row["instantaneous_status"] == "ZERO_DOWNSTREAM_DRIVE"
                  for row in dormant_source["candidate_source_states"]),
    })

    handoff_prefix, _ = deterministic_trajectory(stop_before_ligament=True)
    handoff_connected, _ = ligament_transaction(handoff_prefix)
    handoff_child, _, _, first_handoff = downstream_front_transaction(handoff_connected)
    child_tensor, child_elements = crack_tip_tensor(handoff_child, branch_id="void-front-1")
    _, child_base_audit = _complete_next_clock(
        handoff_child, child_tensor, source_kind="sharp_front",
        source_front_id="void-front-1",
        source_position_m=handoff_child.crack_network.branch("void-front-1").tip,
        source_probe_identity={"kind": "child_crack_tip_tensor",
                               "element_ids": list(child_elements)},
    )
    _, child_raised_audit = _complete_next_clock(
        handoff_child, child_tensor * 1.1, source_kind="sharp_front",
        source_front_id="void-front-1",
        source_position_m=handoff_child.crack_network.branch("void-front-1").tip,
        source_probe_identity={"kind": "child_crack_tip_tensor",
                               "element_ids": list(child_elements)},
    )
    base_rate = child_base_audit[0]["effective_rate_s"]
    raised_rate = child_raised_audit[0]["effective_rate_s"]
    rows.append({
        "gate": "V5_CHILD_TIP_CAUSAL_PERTURBATION",
        "case": "child_tip_field_controls_continuation",
        "initial": observables(handoff_connected, "before_child"),
        "final": observables(handoff_child, "active_child"),
        "first_passage_source_kind": first_handoff["source_kind"],
        "continuation_source_front_id": "void-front-1",
        "child_tensor_Pa": child_tensor.tolist(),
        "child_tensor_element_ids": list(child_elements),
        "base_child_rate_s": base_rate,
        "raised_child_rate_s": raised_rate,
        "obsolete_cavity_probe_is_active_route": False,
        "passed": first_handoff["source_kind"] == "cavity_surface"
                  and raised_rate > base_rate > 0.0
                  and handoff_child.junction_process_state["active_event_source"]["source_front_id"]
                      == "void-front-1",
    })

    tie_state, tie_prefix = deterministic_trajectory(stop_before_ligament=True)
    tie_candidates = (
        CleavageCandidate.create(
            plane_family="cleavage", plane_variant="evidence-hits-cavity",
            direction_xy=(1.0, 0.0), normal_xy=(0.0, 1.0), gamma_rel=1.0,
            orientation_convention="V5 Phase-A distinct-direction evidence",
        ),
        CleavageCandidate.create(
            plane_family="cleavage", plane_variant="evidence-misses-cavity",
            direction_xy=(0.0, 1.0), normal_xy=(1.0, 0.0), gamma_rel=1.0,
            orientation_convention="V5 Phase-A distinct-direction evidence",
        ),
    )
    tie_competition = DirectionalCompetitionState(
        candidates=tie_candidates,
        hazard_states=tuple(DirectionalHazardState(candidate.candidate_id)
                            for candidate in tie_candidates),
        global_hazard_seed=3621,
    )
    tie_probe = replace(tie_state, competition=tie_competition)
    _, tie_audit = _complete_next_clock(
        tie_probe, crack_tip_tensor(tie_probe, branch_id="b00000000")[0]
    )
    tie_rates = {row["candidate_id"]: row["rate_s"] for row in tie_audit}
    tied = replace(tie_competition, hazard_states=tuple(
        replace(hazard, current_threshold_action=tie_rates[hazard.candidate_id])
        for hazard in tie_competition.hazard_states
    ))
    tie_connected, _ = ligament_transaction(replace(tie_state, competition=tied))
    tie_provenance = tie_connected.junction_process_state["directional_event_provenance"]
    consumed = sorted(row["candidate_id"] for row in tie_provenance.values()
                      if row["status"] == "CONSUMED_AT_OWNED_SOURCE")
    stale = sorted(row["candidate_id"] for row in tie_provenance.values()
                   if row["status"] == "COMPLETED_BUT_GEOMETRICALLY_STALE")
    rows.append({
        "gate": "V5_DISTINCT_DIRECTION_TIE_EVIDENCE", "case": "intersecting_winner_stale_miss",
        "initial": tie_prefix[-1], "final": observables(tie_connected, "connected_tie"),
        "intersecting_candidate_id": tie_candidates[0].candidate_id,
        "nonintersecting_candidate_id": tie_candidates[1].candidate_id,
        "consumed_candidate_ids": consumed, "geometrically_stale_candidate_ids": stale,
        "passed": consumed == [tie_candidates[0].candidate_id]
                  and stale == [tie_candidates[1].candidate_id],
    })

    for angle_deg in (10.0, 30.0):
        angle = math.radians(angle_deg)
        oblique_candidate = CleavageCandidate.create(
            plane_family="cleavage", plane_variant=f"fixed-mesh-oblique-{angle_deg:g}",
            direction_xy=(math.cos(angle), math.sin(angle)),
            normal_xy=(-math.sin(angle), math.cos(angle)), gamma_rel=1.0,
            orientation_convention="V5 prospective fixed-mesh oblique qualification",
        )
        if angle_deg == 30.0:
            direction = np.asarray((math.cos(angle), math.sin(angle)))
            center = np.asarray((7.0e-4, 0.0))
            tip = center - 1.5e-4 * direction
            root = np.asarray((0.0, tip[1] - tip[0] * direction[1] / direction[0]))
            oblique_prefix, oblique_rows = deterministic_trajectory(
                stop_before_ligament=True,
                crack_path_m=(tuple(root), tuple(tip)), cleavage_theta_deg=30.0,
            )
            case_name = "30_degree_external_free_root_and_ligament"
        else:
            oblique_prefix, oblique_rows = deterministic_trajectory(stop_before_ligament=True)
            oblique_prefix = replace(oblique_prefix, competition=DirectionalCompetitionState(
                candidates=(oblique_candidate,),
                hazard_states=(DirectionalHazardState(oblique_candidate.candidate_id),),
                global_hazard_seed=3621,
            ))
            case_name = "10_degree_ligament_from_horizontal_external_root"
        oblique_connected, oblique_result = ligament_transaction(oblique_prefix)
        oblique_cavity = oblique_connected.void_state.cavities[0]
        oblique_ledger = oblique_connected.void_state.length_ledgers
        source_rows_oblique = oblique_connected.junction_process_state["active_event_source"][
            "candidate_source_states"
        ]
        rows.append({
            "gate": "V5_OBLIQUE_CONNECTION_EVIDENCE",
            "case": case_name,
            "initial": oblique_rows[-1],
            "final": observables(oblique_connected, "oblique_connected"),
            "angle_deg": angle_deg,
            "connection_entry_m": list(oblique_cavity.connection_entry_m),
            "connection_exit_m": list(oblique_cavity.connection_exit_m),
            "physical_chord_m": math.dist(oblique_cavity.connection_entry_m,
                                           oblique_cavity.connection_exit_m),
            "projected_chord_m": (oblique_cavity.connection_exit_m[0]
                                   - oblique_cavity.connection_entry_m[0]),
            "propagation_side_boundary_arc_m": oblique_ledger["connected_free_surface_extent_m"],
            "candidate_source_states": list(source_rows_oblique),
            "classification": "CONNECTED_VOID_ZERO_DOWNSTREAM_DRIVE"
                              if all(row["instantaneous_status"] == "ZERO_DOWNSTREAM_DRIVE"
                                     for row in source_rows_oblique)
                              else "CONNECTED_VOID_WITH_ACTIVE_DOWNSTREAM_CANDIDATE",
            "topology_certificate": oblique_connected.junction_process_state[
                "latest_crack_void_connection_certificate"
            ],
            "passed": oblique_result.accepted
                      and oblique_connected.junction_process_state[
                          "latest_crack_void_connection_certificate"
                      ]["passed"]
                      and oblique_ledger["connected_void_free_span_m"]
                          == math.dist(oblique_cavity.connection_entry_m,
                                       oblique_cavity.connection_exit_m)
                      and oblique_ledger["projected_connected_void_free_span_m"]
                          == oblique_cavity.connection_exit_m[0] - oblique_cavity.connection_entry_m[0],
        })

    for angle_deg, center_x in ((30.0, 7.0e-4), (45.0, 4.5e-4)):
        angle = math.radians(angle_deg)
        direction = np.asarray((math.cos(angle), math.sin(angle)))
        center = np.asarray((center_x, 0.0))
        tip = center - 1.5e-4 * direction
        root = np.asarray((0.0, tip[1] - tip[0] * direction[1] / direction[0]))
        boundary_state, boundary_rows = deterministic_trajectory(
            stop_before_ligament=True, cavity_center_m=tuple(center),
            crack_path_m=(tuple(root), tuple(tip)), cleavage_theta_deg=angle_deg,
        )
        certificates = boundary_state.junction_process_state[
            "v12_boundary_terminal_certificates"
        ]
        external = next(row for row in certificates
                        if row["boundary_kind"] == "external_free_surface")
        rows.append({
            "gate": "V5_BOUNDARY_RELATIVE_CUT_CERTIFICATE",
            "case": f"physical_external_free_surface_root_{angle_deg:g}_degree",
            "initial": boundary_rows[0], "final": boundary_rows[-1],
            "angle_deg": angle_deg, "boundary_certificate": external,
            "support_fingerprint": boundary_state.junction_process_state[
                "v12_graph_support_audit"
            ]["support_fingerprint"],
            "certificate_fingerprint": boundary_state.junction_process_state[
                "v12_graph_support_audit"
            ]["certificate_fingerprint"],
            "passed": external["classification"]
                      == "CERTIFIED_RELATIVE_TO_EXTERNAL_FREE_BOUNDARY"
                      and external["exact_incidence"]
                      and external["tangent_enters_or_approaches_solid"]
                      and external["positive_interior_seed_count"] > 0
                      and external["negative_interior_seed_count"] > 0
                      and not external["intact_path_exists"]
                      and external["node_star_passed"],
        })

    # Prospectively frozen constitutive test: one state, candidate, orientation,
    # threshold and hazard action; only the supplied local tensor changes.
    direct_state, _ = build_production_void_state(enabled=True)
    fixed_candidate_id = direct_state.competition.candidates[0].candidate_id
    perturbation_rates = []
    for opening_Pa in (2.0e8, 4.0e8, 6.0e8, 8.0e8):
        tensor = [[opening_Pa, 0.0], [0.0, opening_Pa]]
        _, audit = _complete_next_clock(direct_state, tensor)
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
        _, audit = _complete_next_clock(connected, tensor)
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
        "gate": "V5_REFERENCE_LIFECYCLE_SCREEN", "case": "production_checkpoint_restart",
        "initial": observables(final, "checkpointed"), "final": observables(restored, "restored"),
        "checkpoint_manifest": manifest,
        "passed": complete_accepted_state_fingerprint(final) == complete_accepted_state_fingerprint(restored),
    })

    natural_final, natural_rows = natural_trajectory()
    natural_events = [event for row in natural_rows for event in row["events"]]
    rows.append({
        "gate": "V5_NATURAL_WINDOW_ATTEMPTED", "case": "actual_stress_history_seed_3621",
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
                "exact_no_intact_node_or_element_path": {"source_row_id": f"raw:{index}", "path": base + ["exact_no_intact_node_or_element_path"]},
                "exact_triangle_polygon_interior_overlap_absent": {"source_row_id": f"raw:{index}", "path": base + ["exact_triangle_polygon_interior_overlap_absent"]},
                "combined_incidence_component_count": {"source_row_id": f"raw:{index}", "path": base + ["combined_incidence_component_count"]},
                "bridge_search_uncovered_sample_indices": {"source_row_id": f"raw:{index}", "path": base + ["bridge_search_uncovered_sample_indices"]},
                "support_triangle_cavity_overlap_element_ids": {"source_row_id": f"raw:{index}", "path": base + ["support_triangle_cavity_overlap_element_ids"]},
                "crack_segment_cavity_intersections": {"source_row_id": f"raw:{index}", "path": base + ["crack_segment_cavity_intersections"]},
                "combined_components": {"source_row_id": f"raw:{index}", "path": base + ["combined_components"]},
                "combined_incidence_edges": {"source_row_id": f"raw:{index}", "path": base + ["combined_incidence_edges"]},
            }}
        elif row["case"] == "connected_dormancy_then_downstream_sole_activation":
            evidence["predicate_name"] = "single_active_front_ownership"
            evidence["predicate_inputs"] = {"source_bindings": {
                name: {"source_row_id": f"raw:{index}", "path": [name]}
                for name in ("connected_graph_active_ids", "connected_support_active_ids",
                             "downstream_graph_active_ids", "downstream_support_active_ids")
            }}
        elif row["case"] == "offset_pair_symmetry":
            evidence["predicate_name"] = "offset_pair_symmetry"
            names = ("positive_certified", "negative_certified", "reaction_relative_error",
                     "reaction_tolerance", "compliance_relative_error", "compliance_tolerance",
                     "mirrored_intersection_error_m", "geometry_tolerance_m")
            evidence["predicate_inputs"] = {"source_bindings": {
                name: {"source_row_id": f"raw:{index}", "path": [name]} for name in names
            }}
        elif row["case"] == "complete_reference_length_identities":
            evidence["predicate_name"] = "generalized_length_identities"
            names = ("physical_front_travel_m", "physical_fractured_m",
                     "traversed_physical_void_m", "projected_front_advance_m",
                     "projected_fractured_m", "projected_void_m", "tolerance_m")
            evidence["predicate_inputs"] = {"source_bindings": {
                name: {"source_row_id": f"raw:{index}", "path": [name]} for name in names
            }}
        elif row["case"] == "cavity_to_child_tip_causal_handoff":
            evidence["predicate_name"] = "child_sharp_front_handoff"
            names = ("first_source_kind", "continued_source_kind", "continued_source_front_id",
                     "active_branch_id", "r_tip_m", "topology_stage")
            evidence["predicate_inputs"] = {"source_bindings": {
                name: {"source_row_id": f"raw:{index}", "path": [name]} for name in names
            }}
        elif row["case"] == "terminal_equilibrium_decomposition":
            evidence["predicate_name"] = "free_equilibrium_metrics"
            names = ("free_residual_l2", "constrained_reaction_l2", "reaction_balance",
                     "energy_reaction_identity", "free_residual_relative_tolerance",
                     "reaction_balance_tolerance", "energy_identity_tolerance")
            evidence["predicate_inputs"] = {"source_bindings": {
                name: {"source_row_id": f"raw:{index}", "path": [name]} for name in names
            }}
        elif row["case"] == "intersecting_winner_stale_miss":
            evidence["predicate_name"] = "distinct_direction_source_eligibility"
            names = ("intersecting_candidate_id", "nonintersecting_candidate_id",
                     "consumed_candidate_ids", "geometrically_stale_candidate_ids")
            evidence["predicate_inputs"] = {"source_bindings": {
                name: {"source_row_id": f"raw:{index}", "path": [name]} for name in names
            }}
        elif row["case"] == "negative_offset_connected_zero_drive_partitions":
            evidence["predicate_name"] = "zero_drive_connected_invariance"
            names = ("classification", "active_graph_tip_ids", "active_support_tip_ids",
                     "child_branch_exists", "partition_measurements")
            evidence["predicate_inputs"] = {"source_bindings": {
                name: {"source_row_id": f"raw:{index}", "path": [name]} for name in names
            }}
        elif row["case"].startswith("topology_"):
            evidence["predicate_name"] = "eventwise_exact_topology"
            evidence["predicate_inputs"] = {"source_bindings": {
                "certificate_passed": {"source_row_id": f"raw:{index}",
                                       "path": ["certificate", "passed"]},
                "exact_no_intact_path": {"source_row_id": f"raw:{index}",
                                         "path": ["certificate", "exact_no_intact_node_or_element_path"]},
                "exact_polygon_overlap_absent": {"source_row_id": f"raw:{index}",
                                                  "path": ["certificate", "exact_triangle_polygon_interior_overlap_absent"]},
                "component_members": {"source_row_id": f"raw:{index}",
                                      "path": ["certificate", "combined_components", 0]},
                "expected_component_members": {"source_row_id": f"raw:{index}",
                                                "path": ["expected_component_members"]},
            }}
        elif row["case"] == "child_tip_field_controls_continuation":
            evidence["predicate_name"] = "child_tip_causal_perturbation"
            names = ("first_passage_source_kind", "continuation_source_front_id",
                     "base_child_rate_s", "raised_child_rate_s",
                     "obsolete_cavity_probe_is_active_route")
            evidence["predicate_inputs"] = {"source_bindings": {
                name: {"source_row_id": f"raw:{index}", "path": [name]} for name in names
            }}
        elif row["gate"] == "V5_OBLIQUE_CONNECTION_EVIDENCE":
            evidence["predicate_name"] = "oblique_connection_geometry"
            evidence["predicate_inputs"] = {"source_bindings": {
                "topology_passed": {"source_row_id": f"raw:{index}",
                                    "path": ["topology_certificate", "passed"]},
                "physical_chord_m": {"source_row_id": f"raw:{index}", "path": ["physical_chord_m"]},
                "projected_chord_m": {"source_row_id": f"raw:{index}", "path": ["projected_chord_m"]},
                "propagation_side_boundary_arc_m": {"source_row_id": f"raw:{index}",
                                                     "path": ["propagation_side_boundary_arc_m"]},
                "classification": {"source_row_id": f"raw:{index}", "path": ["classification"]},
            }}
        elif row["gate"] == "V5_BOUNDARY_RELATIVE_CUT_CERTIFICATE":
            evidence["predicate_name"] = "boundary_relative_cut_certificate"
            base = ["boundary_certificate"]
            evidence["predicate_inputs"] = {
                "expected_classification": "CERTIFIED_RELATIVE_TO_EXTERNAL_FREE_BOUNDARY",
                "source_bindings": {
                    "classification": {"source_row_id": f"raw:{index}", "path": base + ["classification"]},
                    "exact_incidence": {"source_row_id": f"raw:{index}", "path": base + ["exact_incidence"]},
                    "tangent_enters_solid": {"source_row_id": f"raw:{index}", "path": base + ["tangent_enters_or_approaches_solid"]},
                    "positive_seed_count": {"source_row_id": f"raw:{index}", "path": base + ["positive_interior_seed_count"]},
                    "negative_seed_count": {"source_row_id": f"raw:{index}", "path": base + ["negative_interior_seed_count"]},
                    "intact_path_exists": {"source_row_id": f"raw:{index}", "path": base + ["intact_path_exists"]},
                    "node_star_passed": {"source_row_id": f"raw:{index}", "path": base + ["node_star_passed"]},
                    "only_boundary_clearance_waived": {"source_row_id": f"raw:{index}", "path": base + ["only_boundary_clearance_waived"]},
                },
            }
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
        "broad_qualification_gates": {
            "V12_ONE_VOID_END_TO_END_DEMONSTRATED": "NOT_RUN",
            "V12_VOID_LIFECYCLE_QUALIFIED": "OPEN",
            "V12_VOID_PROMOTION_AND_GROWTH_QUALIFIED": "OPEN",
            "V12_CRACK_VOID_TRANSACTION_QUALIFIED": "OPEN",
            "V12_BOUNDED_NATURAL_STOCHASTIC_CASE": "NOT_RUN",
        },
        "multiple_voids_enabled": False, "fatigue_campaign_run": False,
    }
    path = out / "case_rows.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (out / "sha256_manifest.json").write_text(json.dumps({"case_rows.json": hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2) + "\n")
    print(json.dumps(gates, indent=2, sort_keys=True))
    return 0 if all(value == "PASS" for value in gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
