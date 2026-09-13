import json
from pathlib import Path

import numpy as np
import pytest

from arrhenius_fracture.cavity_source_conforming_geometry_v4 import source_node_certificate
from arrhenius_fracture.cavity_source_shape_regular_mesh_v5 import (
    LOCAL_LEVELS, MESH_CONTRACT_ID, build_shape_regular_source_patch_hole_mesh,
)

ROOT = Path(__file__).resolve().parents[1]


def test_v4_mesh_failure_audit_is_geometry_only_and_locates_crack_support_cause():
    record = json.loads(
        (ROOT / "artifacts/v5_cavity_source_recovery_v5/v4_mesh_failure_audit.json").read_text()
    )
    assert record["new_fem_solve_run"] is False
    assert record["cause_decision"]["classification"] == "C_CRACK_SUPPORT_CAVITY_INTERACTION"
    evidence = record["cause_decision"]["evidence"]
    assert evidence["N128_base_mesh_quality_above_acceptance"] is True
    assert evidence["N128_quality_drops_below_acceptance_after_fixed_crack_path_insertion"] is True
    assert evidence["N128_worst_element_location"] == "crack_support_region"
    assert evidence["retained_connected_mesh_collapses_further_in_same_construction_path"] is True
    assert len(record["levels"]) == 3
    assert all(len(level["lowest_quality_elements"]) == 20 for level in record["levels"])
    retained = record["retained_v4_interpretation"]
    assert retained["V4_SOURCE_GEOMETRY"] == "PASS"
    assert retained["V4_POINT_SOURCE_FORMULATION"] == "REMAINS_VIABLE"
    assert retained["FINITE_ACTIVATION_ZONE_REQUIRED"] == "NOT_ESTABLISHED"
    assert retained["V4_SOURCE_TENSOR_RELATIVE_CHANGE_64_TO_128"] == 0.006818275586047867
    assert "not an independent raw adjacent-element traction" in retained[
        "zero_recovered_boundary_traction_interpretation"
    ]


def _hole(level, sectors=128):
    return build_shape_regular_source_patch_hole_mesh(
        1.0e-3, 1.0e-3, (7.0e-4, 0.0), 5.5e-5, 5.0e-5,
        local_level=level, polygon_sectors=sectors,
    )


@pytest.mark.parametrize("level", ("A", "B", "C"))
def test_shape_regular_levels_preserve_exact_boundary_and_source_frame(level):
    hole = _hole(level)
    assert hole.validation["geometry_contract"] == MESH_CONTRACT_ID
    assert hole.validation["cavity_cycle"] is True
    assert hole.validation["minimum_quality"] >= 0.05
    source = np.asarray(hole.center_m) + np.asarray((hole.radius_m, 0.0))
    node = int(np.argmin(np.linalg.norm(hole.mesh.nodes - source, axis=1)))
    certificate = source_node_certificate(
        nodes=hole.mesh.nodes, elements=hole.mesh.elems,
        owned_boundary_edges=hole.cavity_edges, boundary_node=node,
        cavity_center_m=hole.center_m, cavity_radius_m=hole.radius_m,
    )
    assert certificate["degree"] == 2
    assert certificate["radial_error_m"] <= 1e-15
    assert certificate["unique_tangent_and_outward_normal"] is True
    owner_counts = {}
    for tri in hole.mesh.elems:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = tuple(sorted((int(a), int(b))))
            owner_counts[edge] = owner_counts.get(edge, 0) + 1
    assert all(owner_counts[tuple(sorted(map(int, edge)))] == 1
               for edge in hole.cavity_edges)


def test_local_levels_hold_polygon_and_discrete_boundary_fingerprints_fixed():
    holes = [_hole(level) for level in ("A", "B", "C")]
    assert len({hole.validation["polygon_geometry_fingerprint"] for hole in holes}) == 1
    assert len({hole.validation["discrete_cavity_boundary_fingerprint"] for hole in holes}) == 1
    assert [hole.validation["first_strip_radial_subdivisions"] for hole in holes] == [2, 4, 8]


def test_angular_family_has_matched_boundary_and_radial_source_resolution():
    coarse, fine = _hole("C", 64), _hole("C", 128)
    assert len(coarse.cavity_edges) == len(fine.cavity_edges) == 512
    assert coarse.validation["first_strip_radial_subdivisions"] == 16
    assert fine.validation["first_strip_radial_subdivisions"] == 8
    assert coarse.validation["minimum_quality"] >= 0.05
    assert fine.validation["minimum_quality"] >= 0.05


def test_v5_contract_is_prospective_and_records_measured_pre_fem_quality():
    from scripts.build_cavity_source_shape_regular_v5_contract import OUTPUT, build_record

    retained = json.loads(OUTPUT.read_text())
    regenerated = build_record()
    assert regenerated["mesh_contract"] == retained["mesh_contract"] == MESH_CONTRACT_ID
    assert retained["prospectively_frozen_before_central_dbtt_evaluation"] is True
    assert [row["level"] for row in retained["local_levels"]] == ["A", "B", "C"]
    assert all(row["pre_fem_minimum_quality"] >= 0.05 for row in retained["local_levels"])
    assert len({row["polygon_geometry_fingerprint"] for row in retained["local_levels"]}) == 1
    assert len({row["discrete_cavity_boundary_fingerprint"] for row in retained["local_levels"]}) == 1
    assert retained["scope"]["finite_activation_zone_observable_derived"] is False


def _readiness():
    return json.loads(
        (ROOT / "artifacts/v5_cavity_source_recovery_v5/central_dbtt_v5_readiness.json").read_text()
    )


def test_central_dbtt_v5_fails_only_the_independent_raw_traction_gate():
    record = _readiness()
    assert record["DBTT_SOURCE_READINESS"] == "BLOCKED_WITH_EXACT_V5_FAILURE_CLASS"
    assert record["exact_v5_failure_class"] == ["RAW_ADJACENT_ELEMENT_TRACTION"]
    angular = record["angular_family"]["required_64_to_128"]
    assert angular["predicates"]["source_tensor_convergence"] is True
    assert angular["predicates"]["minimum_mesh_quality"] is True
    assert angular["predicates"]["exact_material_load_threshold_rng_source_identity"] is True
    assert angular["predicates"]["raw_adjacent_traction"] is False
    local = record["fixed_geometry_local_family"]["B_to_C"]
    assert {key for key, passed in local["predicates"].items() if not passed} == {
        "raw_adjacent_traction"
    }


def test_v5_source_diagnostics_keep_constrained_raw_and_weak_residuals_separate():
    rows = _readiness()["fixed_geometry_local_family"]["rows"]
    assert [row["local_level"] for row in rows] == ["A", "B", "C"]
    assert all(row["constrained_boundary_limit_traction_residual"] == 0.0 for row in rows)
    assert rows[1]["raw_adjacent_element_traction_normalized"] > 0.05
    assert rows[2]["raw_adjacent_element_traction_normalized"] > 0.05
    assert all(row["assembled_weak_cavity_boundary_residual_normalized"] > 0.0 for row in rows)
    assert all(row["sigma_zz_and_mean_stress_audit"]["classification"] == "DIAGNOSTIC_ONLY"
               for row in rows)
    assert all(row["sigma_zz_and_mean_stress_audit"][
        "mean_stress_consumed_by_cavity_source_kinetics"
    ] is False for row in rows)


def test_v5_fixed_source_window_identity_and_scope_are_preserved():
    record = _readiness()
    rows = record["fixed_geometry_local_family"]["rows"]
    assert len({row["physical_window_identity"] for row in rows}) == 1
    assert len({
        row["discrete_cavity_boundary_fingerprints"]["fixed_v3_source_window"]
        for row in rows
    }) == 1
    assert all(row["accepted_input_state_unchanged"] for row in rows)
    conditional = record["angular_family"]["conditional_256_row"]
    assert conditional["N_theta"] == 256
    assert conditional["global_minimum_quality"] < 0.05
    assert record["oracle_states_accepted"] == 0
    assert record["oracle_generated_in_this_record"] is False
    assert record["paired_trajectories_run"] == 0
    assert record["fatigue_started"] is False
    assert record["scope"]["finite_activation_zone_observable_derived"] is False
    assert record["scope"]["r_tip_law_changed"] is False
    assert record["scope"]["r_tip_equals_R_void"] is False
