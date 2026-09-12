import json
import math
import pickle
from pathlib import Path

import numpy as np
import pytest

from arrhenius_fracture.cavity_source_conforming_geometry_v4 import (
    GEOMETRY_ID,
    radius_convention_from_hole,
    recover_source_conforming_geometry_v4,
    source_node_certificate,
)
from arrhenius_fracture.explicit_cavity_v5 import (
    build_explicit_hole_mesh,
    build_source_conforming_hole_mesh,
)


ROOT = Path(__file__).resolve().parents[1]
IDENTITY = {
    "core_model_id": "test-core",
    "material_bundle_id": "test-bundle",
    "material_bundle_sha256": "bundle-sha",
    "elasticity_fingerprint": "elastic-sha",
    "plasticity_fingerprint": "plastic-sha",
    "FrontConfig_fingerprint": "front-sha",
    "cleavage_barrier_fingerprint": "cleavage-sha",
    "emission_barrier_fingerprint": "emission-sha",
    "process_zone_fingerprint": "process-sha",
}


def hole(*, sectors=32, layers=12, direction=(1.0, 0.0)):
    return build_source_conforming_hole_mesh(
        1.0e-3, 1.0e-3, (7.0e-4, 0.0), 5.5e-5, 5.0e-5, sectors,
        radial_layers_override=layers, source_direction_xy=direction,
    )


def polygon_measure(value):
    ids = np.asarray(value.prescribed_polygon_nodes, dtype=int)
    points = value.mesh.nodes[ids]
    following = np.roll(points, -1, axis=0)
    return (
        0.5 * abs(float(np.sum(points[:, 0] * following[:, 1]
                               - points[:, 1] * following[:, 0]))),
        float(np.sum(np.linalg.norm(following - points, axis=1))),
    )


def source_node(value, sign=1.0, direction=(1.0, 0.0)):
    vector = np.asarray(direction, dtype=float)
    vector /= np.linalg.norm(vector)
    point = np.asarray(value.center_m) + sign * value.radius_m * vector
    node = int(np.argmin(np.linalg.norm(value.mesh.nodes - point, axis=1)))
    assert value.mesh.nodes[node] == pytest.approx(point, abs=1.0e-15)
    return node


def kirsch_tensor(point, *, center=(7.0e-4, 0.0), radius=5.5e-5, remote=1.0e9):
    x, y = np.asarray(point) - np.asarray(center)
    r = math.hypot(x, y)
    theta = math.atan2(y, x)
    ratio2 = (radius / r) ** 2
    ratio4 = ratio2**2
    c2, s2 = math.cos(2.0 * theta), math.sin(2.0 * theta)
    rr = 0.5 * remote * (1.0 - ratio2) + 0.5 * remote * (
        1.0 - 4.0 * ratio2 + 3.0 * ratio4
    ) * c2
    tt = 0.5 * remote * (1.0 + ratio2) - 0.5 * remote * (
        1.0 + 3.0 * ratio4
    ) * c2
    rt = -0.5 * remote * (1.0 + 2.0 * ratio2 - 3.0 * ratio4) * s2
    normal = np.asarray((math.cos(theta), math.sin(theta)))
    tangent = np.asarray((-math.sin(theta), math.cos(theta)))
    return rr * np.outer(normal, normal) + tt * np.outer(tangent, tangent) + rt * (
        np.outer(normal, tangent) + np.outer(tangent, normal)
    )


def test_retained_v3_polygon_vertex_mismatch_is_analytic_and_frozen():
    record = json.loads(
        (ROOT / "artifacts/v5_cavity_source_recovery_v4/v3_geometry_diagnosis.json").read_text()
    )
    radius = 5.5e-5
    sectors = 32
    vertex_radius = radius / math.cos(math.pi / sectors)
    assert vertex_radius == pytest.approx(5.526612148069713e-5, abs=1.0e-20)
    assert vertex_radius - radius == pytest.approx(2.6612148069713e-7, abs=1.0e-20)
    assert record["R_vertex_m"] == pytest.approx(vertex_radius, abs=1.0e-20)
    assert record["R_vertex_minus_R_void_m"] == pytest.approx(
        vertex_radius - radius, abs=1.0e-20
    )
    assert record["classifications"] == {
        "V3_OPERATOR_MANUFACTURED_AND_KIRSCH": "PASS",
        "V3_CENTRAL_DBTT_GEOMETRY_REGISTRATION": (
            "FAIL_POLYGON_VERTEX_VS_NOMINAL_CIRCLE"
        ),
        "V3_CENTRAL_DBTT_TENSOR_CONVERGENCE": "NOT_RUN",
    }
    assert record["retained_v3_files_modified"] is False


def test_source_conforming_cycle_has_one_owner_and_preserves_polygon_measure():
    legacy = build_explicit_hole_mesh(
        1.0e-3, 1.0e-3, (7.0e-4, 0.0), 5.5e-5, 5.0e-5, 32,
        radial_layers_override=12, angular_phase_rad=-math.pi / 32,
    )
    conforming = hole()
    assert conforming.validation["geometry_contract"] == GEOMETRY_ID
    assert conforming.validation["cavity_cycle"] is True
    assert len(conforming.cavity_edges) == 34
    assert polygon_measure(conforming) == pytest.approx(polygon_measure(legacy), rel=1e-14)
    owners = []
    for edge in conforming.cavity_edges:
        owners.append(sum(set(map(int, edge)).issubset(set(map(int, tri)))
                          for tri in conforming.mesh.elems))
    assert owners == [1] * len(owners)


def test_rotation_preserves_cavity_area_perimeter_and_nominal_inventory_convention():
    base = hole(direction=(1.0, 0.0))
    rotated = hole(direction=(math.cos(0.37), math.sin(0.37)))
    assert polygon_measure(rotated) == pytest.approx(polygon_measure(base), rel=2e-14)
    record = radius_convention_from_hole(rotated, source_direction_xy=(math.cos(0.37), math.sin(0.37))).as_record()
    assert record["nominal_circle_radius_m"] == 5.5e-5
    assert record["polygon_apothem_m"] == 5.5e-5
    assert record["polygon_circumradius_m"] == pytest.approx(5.5e-5 / math.cos(math.pi / 32))
    assert record["nominal_circle_area_m2"] == pytest.approx(math.pi * (5.5e-5) ** 2)
    assert record["void_kinetics_radius_convention"] == "nominal_physical_circle_radius"
    assert record["length_ledger_boundary_convention"] == (
        "actual_source_conforming_polygon_intersections"
    )


@pytest.mark.parametrize("sign", (-1.0, 1.0))
def test_near_and_far_source_nodes_are_exact_degree_two_collinear_midpoints(sign):
    value = hole()
    node = source_node(value, sign)
    certificate = source_node_certificate(
        nodes=value.mesh.nodes, elements=value.mesh.elems,
        owned_boundary_edges=value.cavity_edges, boundary_node=node,
        cavity_center_m=value.center_m, cavity_radius_m=value.radius_m,
    )
    assert certificate["degree"] == 2
    assert certificate["radial_error_m"] <= 1.0e-15
    assert certificate["collinearity_residual"] <= 1.0e-14
    assert certificate["unique_tangent_and_outward_normal"] is True


def test_reversed_edge_ordering_preserves_source_certificate():
    value = hole()
    node = source_node(value)
    forward = source_node_certificate(
        nodes=value.mesh.nodes, elements=value.mesh.elems,
        owned_boundary_edges=value.cavity_edges, boundary_node=node,
        cavity_center_m=value.center_m, cavity_radius_m=value.radius_m,
    )
    reverse = source_node_certificate(
        nodes=value.mesh.nodes, elements=value.mesh.elems,
        owned_boundary_edges=value.cavity_edges[::-1, ::-1], boundary_node=node,
        cavity_center_m=value.center_m, cavity_radius_m=value.radius_m,
    )
    assert reverse == forward


def test_reflected_source_pair_has_reflected_unique_frames():
    direction = np.asarray((math.cos(0.31), math.sin(0.31)))
    reflected = np.asarray((direction[0], -direction[1]))
    upper = hole(direction=direction)
    lower = hole(direction=reflected)
    upper_record = source_node_certificate(
        nodes=upper.mesh.nodes, elements=upper.mesh.elems,
        owned_boundary_edges=upper.cavity_edges, boundary_node=source_node(upper, 1.0, direction),
        cavity_center_m=upper.center_m, cavity_radius_m=upper.radius_m,
    )
    lower_record = source_node_certificate(
        nodes=lower.mesh.nodes, elements=lower.mesh.elems,
        owned_boundary_edges=lower.cavity_edges, boundary_node=source_node(lower, 1.0, reflected),
        cavity_center_m=lower.center_m, cavity_radius_m=lower.radius_m,
    )
    assert lower_record["normal_xy"] == pytest.approx(
        (upper_record["normal_xy"][0], -upper_record["normal_xy"][1]), abs=1e-14
    )


@pytest.mark.parametrize("sectors,layers", ((32, 12), (64, 24), (128, 48)))
def test_kirsch_recovery_uses_source_conforming_facet(sectors, layers):
    value = hole(sectors=sectors, layers=layers, direction=(0.0, 1.0))
    centroids = value.mesh.nodes[value.mesh.elems].mean(axis=1)
    tensors = np.asarray([kirsch_tensor(point) for point in centroids])
    stress = np.column_stack((tensors[:, 0, 0], tensors[:, 1, 1], tensors[:, 0, 1]))
    record = recover_source_conforming_geometry_v4(
        nodes=value.mesh.nodes, elements=value.mesh.elems, stress_Pa=stress,
        boundary_node=source_node(value, direction=(0.0, 1.0)), cavity_id="test-cavity",
        cavity_center_m=value.center_m, cavity_radius_m=value.radius_m,
        process_length_m=value.radius_m, poisson_ratio=0.3,
        owned_boundary_edges=value.cavity_edges, material_fingerprints=IDENTITY,
        state_fingerprint="accepted-state-sha",
    )
    expected = np.asarray(((3.0e9, 0.0), (0.0, 0.0)))
    error = np.linalg.norm(np.asarray(record["tensor_Pa"]) - expected) / np.linalg.norm(expected)
    assert error <= 0.05
    assert record["traction_residual"] <= 0.05
    assert record["minimum_element_quality"] >= 0.05
    assert record["condition"]["maximum"] <= record["condition"]["limit"]
    assert record["v3_fixed_physical_window_wls_reused_unchanged"] is True


def test_source_boundary_checkpoint_roundtrip_preserves_identity():
    value = hole()
    restored = pickle.loads(pickle.dumps(value))
    assert radius_convention_from_hole(restored).as_record() == radius_convention_from_hole(value).as_record()
    assert source_node_certificate(
        nodes=restored.mesh.nodes, elements=restored.mesh.elems,
        owned_boundary_edges=restored.cavity_edges, boundary_node=source_node(restored),
        cavity_center_m=restored.center_m, cavity_radius_m=restored.radius_m,
    ) == source_node_certificate(
        nodes=value.mesh.nodes, elements=value.mesh.elems,
        owned_boundary_edges=value.cavity_edges, boundary_node=source_node(value),
        cavity_center_m=value.center_m, cavity_radius_m=value.radius_m,
    )


def test_failure_after_first_source_insertion_rolls_back_caller_owned_geometry():
    retained = hole()
    before_nodes = retained.mesh.nodes.copy()
    before_elements = retained.mesh.elems.copy()

    def fail(stage, _current):
        if stage == "source_node_inserted:far":
            raise RuntimeError("injected:source_node_inserted:far")

    with pytest.raises(RuntimeError, match="injected:source_node_inserted:far"):
        build_source_conforming_hole_mesh(
            1.0e-3, 1.0e-3, (7.0e-4, 0.0), 5.5e-5, 5.0e-5, 32,
            radial_layers_override=12, failure_injector=fail,
        )
    np.testing.assert_array_equal(retained.mesh.nodes, before_nodes)
    np.testing.assert_array_equal(retained.mesh.elems, before_elements)


def test_prospective_v4_contract_artifact_is_regenerated_exactly():
    from scripts.build_cavity_source_geometry_v4_contract import OUTPUT, build_record

    retained = json.loads(OUTPUT.read_text())
    regenerated = json.loads(json.dumps(build_record(), sort_keys=True))
    assert regenerated["geometry_contract"] == retained["geometry_contract"]
    assert regenerated["acceptance_gates"] == retained["acceptance_gates"]
    assert regenerated["source_boundary_construction"] == retained["source_boundary_construction"]
    assert regenerated["radius_convention"]["nominal_circle_radius_m"] == retained[
        "radius_convention"
    ]["nominal_circle_radius_m"]
    assert regenerated["radius_convention"]["polygon_circumradius_m"] == pytest.approx(
        retained["radius_convention"]["polygon_circumradius_m"], rel=1e-14
    )
    assert regenerated["radius_convention"]["finite_element_polygon_area_m2"] == pytest.approx(
        retained["radius_convention"]["finite_element_polygon_area_m2"], rel=1e-14
    )
    assert regenerated["radius_convention"]["finite_element_polygon_perimeter_m"] == pytest.approx(
        retained["radius_convention"]["finite_element_polygon_perimeter_m"], rel=1e-14
    )
    assert retained["geometry_contract"] == GEOMETRY_ID
    assert retained["prospectively_frozen_before_central_dbtt_evaluation"] is True
    assert retained["acceptance_gates"] == {
        "bounded_patch_conditioning": True,
        "eta_n_max": 0.03,
        "eta_t_max": 0.025,
        "maximum_angular_levels": 3,
        "minimum_mesh_quality": 0.05,
        "normalized_cavity_traction_max": 0.05,
        "tensor_relative_max": 0.05,
        "angular_levels": [32, 64, 128],
    }


def test_central_dbtt_v4_record_preserves_exact_bounded_failure():
    record = json.loads(
        (ROOT / "artifacts/v5_cavity_source_recovery_v4/central_dbtt_v4_readiness.json").read_text()
    )
    assert record["DBTT_SOURCE_READINESS"] == "BLOCKED_WITH_EXACT_V4_FAILURE_CLASS"
    assert record["exact_v4_failure_class"] == [
        "NORMAL_DIRECTION_RESOLUTION",
        "TANGENTIAL_DIRECTION_RESOLUTION",
        "MESH_QUALITY",
    ]
    assert record["levels_run"] == 3
    assert [row["N_theta"] for row in record["level_records"]] == [32, 64, 128]
    final = record["level_records"][-1]
    assert final["tensor_relative_change_from_previous_level"] <= 0.05
    assert final["normalized_cavity_traction"] <= 0.05
    assert final["patch_condition"]["maximum"] <= final["patch_condition"]["limit"]
    assert final["predicates"]["source_geometry"] is True
    assert final["predicates"]["normal_direction_resolution"] is False
    assert final["predicates"]["tangential_direction_resolution"] is False
    assert final["predicates"]["minimum_mesh_quality"] is False
    assert record["accepted_material_identity_exact_across_levels"] is True
    assert record["physical_geometry_exact_across_levels"] is True
    assert record["thresholds_and_rng_exact_across_levels"] is True
    assert record["accepted_input_state_unchanged_on_noncertification"] is True
    assert record["unexpected_programming_exceptions_caught"] is False
    assert record["oracle_states_accepted"] == 0
    assert record["paired_trajectories_run"] == 0
    assert record["fatigue_started"] is False
