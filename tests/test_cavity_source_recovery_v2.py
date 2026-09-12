import math

import numpy as np
import pytest

from arrhenius_fracture.cavity_source_recovery_v2 import (
    CavitySourceRecoveryUnavailable,
    MAXIMUM_DESIGN_CONDITION,
    RECOVERY_ID,
    TENSOR_RELATIVE_TOLERANCE,
    TRACTION_RESIDUAL_TOLERANCE,
    recover_fixed_arc_patch_v2,
)


def annular_mesh(*, sectors=32, radial_layers=4, radius=1.0, growth=0.20):
    angles = 2.0 * np.pi * np.arange(sectors) / sectors
    radii = radius * (1.0 + growth * np.arange(radial_layers + 1))
    nodes = np.asarray([(r * np.cos(a), r * np.sin(a)) for r in radii for a in angles])
    elements = []
    for layer in range(radial_layers):
        for index in range(sectors):
            following = (index + 1) % sectors
            a = layer * sectors + index
            b = layer * sectors + following
            c = (layer + 1) * sectors + index
            d = (layer + 1) * sectors + following
            elements.extend(((a, d, c), (a, b, d)))
    edges = [(index, (index + 1) % sectors) for index in range(sectors)]
    return nodes, np.asarray(elements, dtype=int), edges


def stress_rows(nodes, elements, field):
    centroids = nodes[elements].mean(axis=1)
    tensors = np.asarray([field(point) for point in centroids])
    return np.column_stack((tensors[:, 0, 0], tensors[:, 1, 1], tensors[:, 0, 1]))


def recover(nodes, elements, edges, stress, *, node=0, center=(0.0, 0.0), radius=1.0):
    return recover_fixed_arc_patch_v2(
        nodes=nodes, elements=elements, stress_Pa=stress,
        boundary_node=node, cavity_id="owned-cavity", cavity_center_m=center,
        cavity_radius_m=radius, owned_boundary_edges=edges,
    )


def test_constant_stress_oracle_and_frozen_contract():
    nodes, elements, edges = annular_mesh()
    expected = np.asarray(((0.0, 0.0), (0.0, 7.5e8)))
    record = recover(nodes, elements, edges, stress_rows(nodes, elements, lambda _: expected))
    assert record["recovery_operator"] == RECOVERY_ID
    assert record["v1_incident_cst_max_principal_status"] == "FAIL_NONCONVERGENT"
    assert record["design_rank"] == 3
    assert record["design_condition"] <= MAXIMUM_DESIGN_CONDITION
    assert record["traction_residual"] <= 1.0e-12
    np.testing.assert_allclose(record["tensor_Pa"], expected, rtol=1.0e-12, atol=1.0e-6)


def test_affine_stress_oracle_recovers_solid_side_boundary_limit():
    nodes, elements, edges = annular_mesh()
    def field(point):
        x, y = point
        return np.asarray(((2.0e8 * (x - 1.0), 1.0e8 * (x - 1.0)),
                           (1.0e8 * (x - 1.0), 6.0e8 + 3.0e8 * (x - 1.0) + 2.0e8 * y)))
    record = recover(nodes, elements, edges, stress_rows(nodes, elements, field))
    np.testing.assert_allclose(record["raw_boundary_local_tensor_Pa"],
                               ((0.0, 0.0), (0.0, 6.0e8)), rtol=1.0e-12, atol=2.0e-6)
    np.testing.assert_allclose(record["tensor_Pa"], ((0.0, 0.0), (0.0, 6.0e8)),
                               rtol=1.0e-12, atol=2.0e-6)


def test_rigid_rotation_covariance_and_reversed_edge_ordering():
    nodes, elements, edges = annular_mesh()
    base_tensor = np.asarray(((0.0, 0.0), (0.0, 4.0e8)))
    base = recover(nodes, elements, edges, stress_rows(nodes, elements, lambda _: base_tensor))
    angle = 0.37
    rotation = np.asarray(((math.cos(angle), -math.sin(angle)),
                           (math.sin(angle), math.cos(angle))))
    rotated_nodes = nodes @ rotation.T
    rotated_tensor = rotation @ base_tensor @ rotation.T
    rotated = recover(rotated_nodes, elements, list(reversed([(b, a) for a, b in edges])),
                      stress_rows(rotated_nodes, elements, lambda _: rotated_tensor),
                      node=0, center=(0.0, 0.0), radius=1.0)
    np.testing.assert_allclose(rotated["tensor_Pa"],
                               rotation @ np.asarray(base["tensor_Pa"]) @ rotation.T,
                               rtol=1.0e-12, atol=1.0e-6)
    assert rotated["physical_arc_identity"]["owned_boundary_edges"] == base["physical_arc_identity"]["owned_boundary_edges"]


def kirsch_tensor(point, *, radius=1.0, remote=1.0e9):
    x, y = point
    r = math.hypot(x, y)
    theta = math.atan2(y, x)
    ratio2 = (radius / r) ** 2
    ratio4 = ratio2 ** 2
    c2, s2 = math.cos(2.0 * theta), math.sin(2.0 * theta)
    rr = 0.5 * remote * (1.0 - ratio2) + 0.5 * remote * (1.0 - 4.0 * ratio2 + 3.0 * ratio4) * c2
    tt = 0.5 * remote * (1.0 + ratio2) - 0.5 * remote * (1.0 + 3.0 * ratio4) * c2
    rt = -0.5 * remote * (1.0 + 2.0 * ratio2 - 3.0 * ratio4) * s2
    normal = np.asarray((math.cos(theta), math.sin(theta)))
    tangent = np.asarray((-math.sin(theta), math.cos(theta)))
    return rr * np.outer(normal, normal) + tt * np.outer(tangent, tangent) + rt * (
        np.outer(normal, tangent) + np.outer(tangent, normal)
    )


def test_kirsch_family_has_two_quality_valid_fine_levels_and_fixed_arc():
    rows = []
    for sectors in (64, 128, 256):
        nodes, elements, edges = annular_mesh(sectors=sectors, radial_layers=4, growth=2.0 * np.pi / sectors)
        node = sectors // 4
        record = recover(nodes, elements, edges, stress_rows(nodes, elements, kirsch_tensor), node=node)
        expected = np.asarray(((3.0e9, 0.0), (0.0, 0.0)))
        error = np.linalg.norm(np.asarray(record["tensor_Pa"]) - expected) / np.linalg.norm(expected)
        rows.append((record, error))
    for record, error in rows[-2:]:
        assert record["minimum_element_quality"] >= 0.05
        assert record["design_condition"] <= MAXIMUM_DESIGN_CONDITION
        assert error <= TENSOR_RELATIVE_TOLERANCE
        assert record["traction_residual"] <= TRACTION_RESIDUAL_TOLERANCE
        np.testing.assert_allclose(record["physical_arc_identity"]["boundary_position_m"], (0.0, 1.0), atol=2.0e-15)
    assert rows[-1][1] <= rows[-2][1]


def test_reflected_kirsch_pair_recovers_reflected_tensor():
    nodes, elements, edges = annular_mesh(sectors=64, radial_layers=4, growth=2.0 * np.pi / 64)
    top = recover(nodes, elements, edges, stress_rows(nodes, elements, kirsch_tensor), node=16)
    reflection = np.asarray(((1.0, 0.0), (0.0, -1.0)))
    mirrored_nodes = nodes @ reflection.T
    mirrored_stress = stress_rows(mirrored_nodes, elements, kirsch_tensor)
    bottom = recover(mirrored_nodes, elements[:, [0, 2, 1]], edges, mirrored_stress, node=16)
    np.testing.assert_allclose(bottom["tensor_Pa"],
                               reflection @ np.asarray(top["tensor_Pa"]) @ reflection.T,
                               rtol=1.0e-12, atol=1.0e-5)
    assert bottom["traction_residual"] == pytest.approx(top["traction_residual"], rel=1.0e-12)


def test_inadequate_support_fails_closed_as_scientific_unavailability():
    nodes, elements, edges = annular_mesh(sectors=8, radial_layers=1)
    solid = np.zeros(len(elements), dtype=bool)
    solid[0] = True
    with pytest.raises(CavitySourceRecoveryUnavailable):
        recover_fixed_arc_patch_v2(
            nodes=nodes, elements=elements,
            stress_Pa=stress_rows(nodes, elements, lambda _: np.eye(2)),
            boundary_node=0, cavity_id="owned-cavity", cavity_center_m=(0.0, 0.0),
            cavity_radius_m=1.0, owned_boundary_edges=edges, solid_element_mask=solid,
        )


def test_single_central_dbtt_readiness_case_fails_closed_after_three_levels():
    from dataclasses import replace
    from arrhenius_fracture.unified_fracture_material_v5 import material_bundle, require_bound_identity
    from arrhenius_fracture.voiding_production_v5 import (
        _cavity_resolution_binding,
        deterministic_trajectory,
        equilibrate_fixed_load_with_production_fem,
        ligament_transaction,
        refine_downstream_source,
    )

    bundle = material_bundle("DBTT")
    preconnection, _ = deterministic_trajectory(bundle=bundle, stop_before_ligament=True)
    loaded = equilibrate_fixed_load_with_production_fem(
        replace(preconnection, displacement=preconnection.displacement * 2.0)
    )
    connected, ligament = ligament_transaction(loaded)
    accepted_binding = _cavity_resolution_binding(connected)
    accepted_clocks = (connected.competition, connected.rng_state)
    qualified, audit = refine_downstream_source(
        connected,
        max_refinement_levels=3,
        refinement_region="complete_cavity_ring",
        quality_improvement="constrained_v1",
    )
    assert bundle.fracture_material_row_id == "v913_zeroD_sobol_0202500"
    assert dict(require_bound_identity(connected))["material_bundle_id"] == bundle.bundle_id
    assert ligament.accepted and ligament.energy_margin_J_per_m > 0.0
    assert qualified is connected
    assert _cavity_resolution_binding(qualified) == accepted_binding
    assert (qualified.competition, qualified.rng_state) == accepted_clocks
    assert audit["status"] == "SOURCE_TENSOR_UNQUALIFIED"
    assert len(audit["attempts"]) == 3
    final = audit["attempts"][-1]["proof"]
    previous = np.asarray(final["previous_metrics"]["tensor_Pa"], dtype=float)
    current = np.asarray(final["current_metrics"]["tensor_Pa"], dtype=float)
    relative_change = np.linalg.norm(current - previous) / np.linalg.norm(current)
    assert relative_change > TENSOR_RELATIVE_TOLERANCE
    assert final["current_metrics"]["normalized_traction"] <= TRACTION_RESIDUAL_TOLERANCE
    recovery = final["current_metrics"]["recovery_record"]
    assert recovery["recovery_operator"] == RECOVERY_ID
    assert recovery["design_rank"] == 3
    assert recovery["design_condition"] <= MAXIMUM_DESIGN_CONDITION
