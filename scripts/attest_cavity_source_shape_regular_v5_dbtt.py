#!/usr/bin/env python3
"""Run only the prospectively frozen central DBTT V5 mesh families."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.cavity_source_conforming_geometry_v4 import (
    CavitySourceGeometryV4Unavailable,
)
from arrhenius_fracture.cavity_source_shape_regular_mesh_v5 import (
    ACCEPTANCE_MINIMUM_QUALITY, LOCAL_LEVELS, MESH_CONTRACT_ID,
)
from arrhenius_fracture.crack_network_v11 import ROOT_BRANCH_ID
from arrhenius_fracture.finalization_v3_schema import SCIENTIFIC_ACCEPTANCE_TOLERANCES
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
from arrhenius_fracture.unified_fracture_material_v5 import (
    identity_record, material_bundle, require_bound_identity,
)
from arrhenius_fracture.voiding_production_v5 import (
    _actual_cavity_boundary_edges, _cavity_resolution_binding,
    build_production_void_state, cavity_source_resolution_metrics,
    deterministic_trajectory, equilibrate_fixed_load_with_production_fem,
    ligament_transaction, observables,
)


OUTPUT = ROOT / "artifacts/v5_cavity_source_recovery_v5/central_dbtt_v5_readiness.json"
ANGULAR_LEVELS = (64, 128)
CONDITIONAL_ANGULAR_LEVEL = 256
LOCAL_LEVELS_RUN = ("A", "B", "C")
FIXED_LOCAL_POLYGON_SECTORS = 128
RADIAL_LAYERS = 48
TENSOR_LIMIT = 0.05
TRACTION_LIMIT = 0.05
REACTION_LIMIT = SCIENTIFIC_ACCEPTANCE_TOLERANCES["static_mesh_reaction_relative"]
COMPLIANCE_LIMIT = SCIENTIFIC_ACCEPTANCE_TOLERANCES["static_mesh_reaction_relative"]
ENERGY_LIMIT = SCIENTIFIC_ACCEPTANCE_TOLERANCES["static_mesh_energy_relative"]


def _compact_observables(full):
    scalar_keys = (
        "operation", "fingerprint", "mesh_nodes", "mesh_elements", "graph_length_m",
        "reaction_N_per_m", "compliance_m2_per_N", "energy_J_per_m",
        "top_reaction_N_per_m", "bottom_reaction_N_per_m", "applied_opening_m",
        "external_work_J_per_m", "stored_recoverable_energy_J_per_m",
        "plastic_eigenstrain_half_work_J_per_m", "energy_identity_reference_J_per_m",
        "full_residual_including_reactions_N_per_m", "free_dof_residual_l2_N_per_m",
        "constrained_reaction_l2_N_per_m", "top_bottom_reaction_balance",
        "energy_reaction_identity", "void_phase", "site_phase", "cavity_radius_m",
        "cavity_area_m2", "inventory_area_m2", "available_defect_inventory_area_m2",
        "consumed_defect_inventory_area_m2", "length_ledgers", "event_counters",
        "active_crack_branch_ids", "support_active_tip_ids", "mesh_minimum_quality",
        "mesh_maximum_aspect_ratio",
    )
    compact = {key: full.get(key) for key in scalar_keys}
    cycle = full.get("closed_cavity_boundary_cycle_certificate") or {}
    connection = full.get("crack_void_connection_certificate") or {}
    compact["topology"] = {
        "closed_cavity_boundary_cycle_passed": cycle.get("passed"),
        "cavity_boundary_edge_count": cycle.get("boundary_edge_count"),
        "cavity_boundary_node_count": cycle.get("boundary_node_count"),
        "crack_void_connection_passed": connection.get("passed"),
        "connection_entry_m": connection.get("connection_entry_m"),
        "connection_exit_m": connection.get("connection_exit_m"),
    }
    return compact


def _hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _relative(first, second):
    return float(np.linalg.norm(np.asarray(second) - np.asarray(first)) /
                 max(np.linalg.norm(np.asarray(second)), 1.0e-300))


def _threshold_identity(state):
    return {
        "global_hazard_seed": int(state.competition.global_hazard_seed),
        "candidate_thresholds": [
            {
                "candidate_id": item.candidate_id,
                "threshold_process": item.threshold_process,
                "threshold_seed": int(item.threshold_seed),
                "current_threshold_action": float(item.current_threshold_action),
            }
            for item in state.competition.hazard_states
        ],
        "void_thresholds": [
            {
                "site_id": site.site_id,
                "birth": float(site.birth.threshold),
                "stabilization": float(site.stabilization.threshold),
                "healing": float(site.healing.threshold),
            }
            for site in state.void_state.sites
        ],
    }


def _boundary_fingerprints(state):
    nodes = np.asarray(state.mesh.nodes, dtype=float)
    edges = np.asarray(_actual_cavity_boundary_edges(state), dtype=int)
    center = np.asarray(state.void_state.cavities[0].center_m, dtype=float)
    source = np.asarray(state.void_state.cavities[0].connection_exit_m, dtype=float)
    source_normal = (source - center) / np.linalg.norm(source - center)
    radius = float(state.void_state.cavities[0].radius_m)
    half_width = 0.5 * min(radius, 5.0e-5)

    def fingerprint(selected_edges):
        unique = np.unique(selected_edges)
        ordered = sorted(map(int, unique), key=lambda node: math.atan2(
            *(nodes[node] - center)[::-1]
        ))
        local_id = {node: index for index, node in enumerate(ordered)}
        return _hash({
            "coordinates_m": nodes[ordered].tolist(),
            "edges_local": sorted(
                sorted((local_id[int(a)], local_id[int(b)])) for a, b in selected_edges
            ),
        })

    midpoints = nodes[edges].mean(axis=1) - center
    angles = np.arctan2(
        source_normal[0] * midpoints[:, 1] - source_normal[1] * midpoints[:, 0],
        midpoints @ source_normal,
    )
    source_edges = edges[np.abs(radius * angles) <= half_width]
    if len(source_edges) < 2:
        raise ValueError("source window has inadequate discrete cavity boundary support")
    return {
        "complete_boundary": fingerprint(edges),
        "fixed_v3_source_window": fingerprint(source_edges),
        "fixed_v3_source_window_edge_count": int(len(source_edges)),
        "fixed_v3_source_window_half_width_m": half_width,
    }


def _run_level(bundle, retained_path, *, sectors, local_level, reference):
    preconnection, _ = deterministic_trajectory(
        bundle=bundle, stop_before_ligament=True, crack_path_m=retained_path,
        boundary_segments=sectors, radial_layers=RADIAL_LAYERS,
        shape_regular_local_level=local_level,
    )
    loaded = equilibrate_fixed_load_with_production_fem(
        replace(preconnection, displacement=preconnection.displacement * 2.0)
    )
    connected, ligament = ligament_transaction(loaded)
    before = complete_accepted_state_fingerprint(connected)
    binding = _cavity_resolution_binding(connected)
    try:
        metrics = cavity_source_resolution_metrics(
            connected, geometry_contract=MESH_CONTRACT_ID
        )
        scientific_unavailability = None
    except CavitySourceGeometryV4Unavailable as exc:
        metrics = None
        scientific_unavailability = str(exc)
    after = complete_accepted_state_fingerprint(connected)
    cavity = connected.void_state.cavities[0]
    identity = dict(require_bound_identity(connected))
    thresholds = _threshold_identity(connected)
    source_coordinate = list(map(float, cavity.connection_exit_m))
    common = {
        "N_theta": sectors,
        "local_level": local_level,
        "scientific_unavailability": scientific_unavailability,
        "accepted_input_state_fingerprint": before,
        "accepted_input_state_unchanged": before == after,
        "mechanical_binding": binding,
        "material_identity": identity,
        "material_identity_matches_reference": identity == reference["material_identity"],
        "threshold_identity": thresholds,
        "thresholds_match_reference": (
            reference["threshold_identity"] is None
            or thresholds == reference["threshold_identity"]
        ),
        "rng_state_matches_reference": connected.rng_state == reference["rng_state"],
        "load_history_matches_reference": True,
        "source_coordinate_m": source_coordinate,
        "source_coordinate_matches_reference": source_coordinate == reference["source_coordinate_m"],
        "nominal_cavity_geometry_matches_reference": (
            tuple(cavity.center_m), float(cavity.radius_m)
        ) == reference["nominal_cavity_geometry"],
        "discrete_cavity_boundary_fingerprints": _boundary_fingerprints(connected),
        "observables": _compact_observables(
            observables(connected, f"V5_{sectors}_{local_level}_connected")
        ),
        "ligament_energy_gate": {
            "accepted": bool(ligament.accepted),
            "energy_release_J_per_m": float(ligament.energy_release_J_per_m),
            "hazard_dissipation_J_per_m": float(ligament.hazard_dissipation_J_per_m),
            "energy_margin_J_per_m": float(ligament.energy_margin_J_per_m),
        },
        "candidate_kinetic_drive": None,
        "candidate_kinetic_drive_status": "NOT_DEFINED_BEFORE_SOURCE_QUALIFICATION",
    }
    if metrics is None:
        return common
    recovery = metrics["recovery_record"]
    return {
        **common,
        "constrained_boundary_limit_tensor_global_xy_Pa": metrics["source_tensor_global_xy_Pa"],
        "constrained_boundary_limit_tensor_local_nt_Pa": metrics["source_tensor_local_nt_Pa"],
        "raw_adjacent_element_traction_normalized": metrics[
            "raw_adjacent_element_traction_normalized"
        ],
        "assembled_weak_cavity_boundary_residual_normalized": metrics[
            "assembled_weak_cavity_boundary_residual_normalized"
        ],
        "constrained_boundary_limit_traction_residual": metrics[
            "constrained_boundary_limit_traction_residual"
        ],
        "constrained_fit_residual": metrics["constrained_fit_residual"],
        "patch_rank": metrics["patch_rank"],
        "patch_condition": metrics["patch_condition"],
        "eta_n": metrics["source_neighborhood_eta_n_max"],
        "eta_t": metrics["source_neighborhood_eta_t_max"],
        "local_minimum_quality": metrics["local_minimum_quality"],
        "global_minimum_quality": metrics["minimum_quality"],
        "physical_window_identity": recovery["physical_arc_identity"]["sha256"],
        "source_node_certificate": recovery["source_node_certificate"],
        "sigma_zz_and_mean_stress_audit": metrics["sigma_zz_and_mean_stress_audit"],
    }


def _pair_comparison(coarse, fine):
    unavailable = coarse.get("scientific_unavailability") or fine.get("scientific_unavailability")
    if unavailable:
        return {"available": False, "scientific_unavailability": unavailable, "passed": False}
    cobs, fobs = coarse["observables"], fine["observables"]
    comparisons = {
        "source_tensor_relative": _relative(
            coarse["constrained_boundary_limit_tensor_global_xy_Pa"],
            fine["constrained_boundary_limit_tensor_global_xy_Pa"],
        ),
        "reaction_relative": _relative(cobs["reaction_N_per_m"], fobs["reaction_N_per_m"]),
        "compliance_relative": _relative(
            cobs["compliance_m2_per_N"], fobs["compliance_m2_per_N"]
        ),
        "energy_relative": _relative(cobs["energy_J_per_m"], fobs["energy_J_per_m"]),
        "ligament_energy_release_relative": _relative(
            coarse["ligament_energy_gate"]["energy_release_J_per_m"],
            fine["ligament_energy_gate"]["energy_release_J_per_m"],
        ),
    }
    predicates = {
        "source_tensor_convergence": comparisons["source_tensor_relative"] <= TENSOR_LIMIT,
        "raw_adjacent_traction": max(
            coarse["raw_adjacent_element_traction_normalized"],
            fine["raw_adjacent_element_traction_normalized"],
        ) <= TRACTION_LIMIT,
        "reaction_convergence": comparisons["reaction_relative"] <= REACTION_LIMIT,
        "compliance_convergence": comparisons["compliance_relative"] <= COMPLIANCE_LIMIT,
        "energy_convergence": comparisons["energy_relative"] <= ENERGY_LIMIT,
        "minimum_mesh_quality": min(
            coarse["global_minimum_quality"], fine["global_minimum_quality"]
        ) >= ACCEPTANCE_MINIMUM_QUALITY,
        "exact_material_load_threshold_rng_source_identity": all(
            row[key]
            for row in (coarse, fine)
            for key in (
                "material_identity_matches_reference", "thresholds_match_reference",
                "rng_state_matches_reference", "load_history_matches_reference",
                "source_coordinate_matches_reference", "nominal_cavity_geometry_matches_reference",
            )
        ),
    }
    return {
        "available": True, "comparisons": comparisons, "predicates": predicates,
        "passed": all(predicates.values()),
    }


def _failure_classes(angular, local, rows):
    failures = []
    if not angular.get("available"):
        failures.append("ANGULAR_SOURCE_RECOVERY_UNAVAILABLE")
    else:
        names = {
            "source_tensor_convergence": "ANGULAR_SOURCE_TENSOR_CONVERGENCE",
            "raw_adjacent_traction": "RAW_ADJACENT_ELEMENT_TRACTION",
            "reaction_convergence": "ANGULAR_REACTION_CONVERGENCE",
            "compliance_convergence": "ANGULAR_COMPLIANCE_CONVERGENCE",
            "energy_convergence": "ANGULAR_ENERGY_CONVERGENCE",
            "minimum_mesh_quality": "ANGULAR_MESH_QUALITY",
            "exact_material_load_threshold_rng_source_identity": "ANGULAR_STATE_IDENTITY",
        }
        failures.extend(names[key] for key, passed in angular["predicates"].items() if not passed)
    names = {
        "source_tensor_convergence": "LOCAL_SOURCE_TENSOR_CONVERGENCE",
        "raw_adjacent_traction": "RAW_ADJACENT_ELEMENT_TRACTION",
        "normal_direction_resolution": "NORMAL_DIRECTION_RESOLUTION",
        "tangential_direction_resolution": "TANGENTIAL_DIRECTION_RESOLUTION",
        "minimum_mesh_quality": "MESH_QUALITY",
        "patch_conditioning": "PATCH_CONDITIONING",
        "physical_window_identity": "PHYSICAL_WINDOW_IDENTITY",
        "discrete_boundary_identity": "DISCRETE_CAVITY_BOUNDARY_IDENTITY",
        "accepted_state_unchanged": "ACCEPTED_INPUT_STATE_MUTATION",
        "exact_state_identity": "LOCAL_STATE_IDENTITY",
    }
    if not local.get("available"):
        failures.append("LOCAL_SOURCE_RECOVERY_UNAVAILABLE")
    else:
        failures.extend(names[key] for key, passed in local["predicates"].items() if not passed)
    if any(not row["ligament_energy_gate"]["accepted"] for row in rows):
        failures.append("LIGAMENT_EVENT_ACCEPTANCE")
    return sorted(set(failures))


def build_record():
    bundle = material_bundle("DBTT")
    retained_initial, _ = build_production_void_state(bundle=bundle, enabled=True)
    retained_path = tuple(retained_initial.crack_network.branch(ROOT_BRANCH_ID).path)
    retained_thresholds = _threshold_identity(retained_initial)
    reference = {
        "material_identity": dict(require_bound_identity(retained_initial)),
        "threshold_identity": None,
        "rng_state": retained_initial.rng_state,
        "source_coordinate_m": [7.55e-4, 0.0],
        "nominal_cavity_geometry": ((7.0e-4, 0.0), 5.5e-5),
    }
    angular_rows = [
        _run_level(bundle, retained_path, sectors=sectors, local_level="C", reference=reference)
        for sectors in ANGULAR_LEVELS
    ]
    connected_threshold_identity = angular_rows[0]["threshold_identity"]
    for row in angular_rows:
        row["thresholds_match_reference"] = (
            row["threshold_identity"] == connected_threshold_identity
        )
    angular_comparison = _pair_comparison(*angular_rows)
    conditional_row = None
    conditional_comparison = None
    if not angular_comparison["passed"]:
        conditional_row = _run_level(
            bundle, retained_path, sectors=CONDITIONAL_ANGULAR_LEVEL,
            local_level="C", reference=reference,
        )
        conditional_row["thresholds_match_reference"] = (
            conditional_row["threshold_identity"] == connected_threshold_identity
        )
        conditional_comparison = _pair_comparison(angular_rows[-1], conditional_row)

    by_level = {"C": angular_rows[-1]}
    for level in ("A", "B"):
        by_level[level] = _run_level(
            bundle, retained_path, sectors=FIXED_LOCAL_POLYGON_SECTORS,
            local_level=level, reference=reference,
        )
        by_level[level]["thresholds_match_reference"] = (
            by_level[level]["threshold_identity"] == connected_threshold_identity
        )
    local_rows = [by_level[level] for level in LOCAL_LEVELS_RUN]
    local_available = all(row.get("scientific_unavailability") is None for row in local_rows)
    if local_available:
        fine, finest = local_rows[1:]
        local_comparisons = {
            "source_tensor_relative_B_to_C": _relative(
                fine["constrained_boundary_limit_tensor_global_xy_Pa"],
                finest["constrained_boundary_limit_tensor_global_xy_Pa"],
            ),
            "reaction_relative_B_to_C": _relative(
                fine["observables"]["reaction_N_per_m"], finest["observables"]["reaction_N_per_m"]
            ),
            "compliance_relative_B_to_C": _relative(
                fine["observables"]["compliance_m2_per_N"],
                finest["observables"]["compliance_m2_per_N"],
            ),
            "potential_energy_relative_B_to_C": _relative(
                fine["observables"]["energy_J_per_m"], finest["observables"]["energy_J_per_m"]
            ),
            "ligament_energy_release_relative_B_to_C": _relative(
                fine["ligament_energy_gate"]["energy_release_J_per_m"],
                finest["ligament_energy_gate"]["energy_release_J_per_m"],
            ),
        }
        local_predicates = {
            "source_tensor_convergence": local_comparisons[
                "source_tensor_relative_B_to_C"
            ] <= TENSOR_LIMIT,
            "raw_adjacent_traction": max(
                row["raw_adjacent_element_traction_normalized"] for row in (fine, finest)
            ) <= TRACTION_LIMIT,
            "normal_direction_resolution": max(row["eta_n"] for row in (fine, finest)) <= 0.03,
            "tangential_direction_resolution": max(row["eta_t"] for row in (fine, finest)) <= 0.025,
            "minimum_mesh_quality": min(
                row["global_minimum_quality"] for row in (fine, finest)
            ) >= ACCEPTANCE_MINIMUM_QUALITY,
            "patch_conditioning": all(
                row["patch_condition"]["maximum"] <= row["patch_condition"]["limit"]
                for row in (fine, finest)
            ),
            "physical_window_identity": len({
                row["physical_window_identity"] for row in local_rows
            }) == 1,
            "discrete_boundary_identity": len({
                row["discrete_cavity_boundary_fingerprints"]["fixed_v3_source_window"]
                for row in local_rows
            }) == 1,
            "accepted_state_unchanged": all(
                row["accepted_input_state_unchanged"] for row in local_rows
            ),
            "exact_state_identity": all(
                row[key] for row in local_rows for key in (
                    "material_identity_matches_reference", "thresholds_match_reference",
                    "rng_state_matches_reference", "load_history_matches_reference",
                    "source_coordinate_matches_reference", "nominal_cavity_geometry_matches_reference",
                )
            ),
        }
        local_comparison = {
            "available": True, "comparisons": local_comparisons,
            "predicates": local_predicates, "passed": all(local_predicates.values()),
        }
    else:
        local_comparison = {"available": False, "passed": False}

    all_rows = angular_rows + ([conditional_row] if conditional_row else []) + local_rows[:2]
    failures = _failure_classes(angular_comparison, local_comparison, all_rows)
    passed = not failures
    return {
        "schema": "v5.central-dbtt-shape-regular-local-patch-readiness/1",
        "executed_code_sha": os.environ.get("V5_SOURCE_COMMIT") or subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
        ).strip(),
        "mesh_contract": MESH_CONTRACT_ID,
        "material_class": "DBTT",
        "fracture_material_row_id": bundle.fracture_material_row_id,
        "material_identity": dict(identity_record(bundle, retained_initial.material)),
        "retained_physical_crack_path_m": [list(map(float, point)) for point in retained_path],
        "retained_initial_threshold_identity": retained_thresholds,
        "connected_threshold_identity_exact_across_levels": connected_threshold_identity,
        "opening_scale_from_4e-7_m": 2.0,
        "angular_family": {
            "required_rows": angular_rows,
            "required_64_to_128": angular_comparison,
            "conditional_256_row": conditional_row,
            "conditional_128_to_256": conditional_comparison,
        },
        "fixed_geometry_local_family": {
            "polygon_sectors": FIXED_LOCAL_POLYGON_SECTORS,
            "rows": local_rows,
            "B_to_C": local_comparison,
        },
        "exact_v5_failure_class": failures,
        "DBTT_SOURCE_READINESS": (
            "PASS_SHAPE_REGULAR_LOCAL_PATCH_V5"
            if passed else "BLOCKED_WITH_EXACT_V5_FAILURE_CLASS"
        ),
        "oracle_states_accepted": 0,
        "oracle_generated_in_this_record": False,
        "paired_trajectories_run": 0,
        "fatigue_started": False,
        "unexpected_programming_exceptions_caught": False,
        "preserved_v4": {
            "DBTT_SOURCE_READINESS": "BLOCKED_WITH_EXACT_V4_FAILURE_CLASS",
            "V4_FAILURE_CLASS": [
                "NORMAL_DIRECTION_RESOLUTION", "TANGENTIAL_DIRECTION_RESOLUTION", "MESH_QUALITY",
            ],
            "V4_SOURCE_GEOMETRY": "PASS",
            "V4_SOURCE_TENSOR_RELATIVE_CHANGE_64_TO_128": 0.006818275586047867,
            "V4_POINT_SOURCE_FORMULATION": "REMAINS_VIABLE",
            "FINITE_ACTIVATION_ZONE_REQUIRED": "NOT_ESTABLISHED",
            "zero_boundary_traction_interpretation": (
                "traction-free constraint of the recovered boundary-limit tensor; "
                "not an independent raw adjacent-element traction measurement"
            ),
        },
        "preserved_v1_v2_v3": {
            "CAVITY_SOURCE_RECOVERY_V1_INCIDENT_CST_MAX_PRINCIPAL": "FAIL_NONCONVERGENT",
            "CAVITY_FIXED_ARC_PATCH_RECOVERY_V2_MANUFACTURED_AND_KIRSCH": "PASS",
            "V3_OPERATOR_MANUFACTURED_AND_KIRSCH": "PASS",
            "V3_CENTRAL_DBTT_GEOMETRY_REGISTRATION": "FAIL_POLYGON_VERTEX_VS_NOMINAL_CIRCLE",
            "V3_CENTRAL_DBTT_TENSOR_CONVERGENCE": "NOT_RUN",
        },
        "scope": {
            "scientific_tolerances_changed": False,
            "r_tip_law_changed": False,
            "r_tip_equals_R_void": False,
            "finite_activation_zone_observable_derived": False,
            "paired_trajectories_run": False,
            "fatigue_started": False,
            "missing_fields_inferred_or_synthesized": False,
        },
        "next_bounded_step": (
            "GENERATE_18_FIXED_STATE_ORACLE_ROWS_IN_SEPARATE_COMMIT"
            if passed else "STOP_POINT_SOURCE_MESH_DEVELOPMENT"
        ),
    }


def main():
    payload = build_record()
    destination = Path(os.environ.get("V5_READINESS_OUTPUT", str(OUTPUT)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("DBTT_SOURCE_READINESS=" + payload["DBTT_SOURCE_READINESS"])
    print("V5_FAILURE_CLASS=" + "+".join(payload["exact_v5_failure_class"]))
    print("ORACLE_STATES_ACCEPTED=" + str(payload["oracle_states_accepted"]) + "/18")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
