import math

import numpy as np
import pytest

from arrhenius_fracture.cavity_source_recovery_v3 import (
    CavitySourceRecoveryV3Unavailable,
    MAXIMUM_DESIGN_CONDITION,
    RECOVERY_ID,
    TENSOR_RELATIVE_TOLERANCE,
    TRACTION_RESIDUAL_TOLERANCE,
    recover_fixed_physical_arc_patch_v3,
)


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


def local_components(point, *, radius=1.0):
    radial = np.asarray(point, dtype=float)
    distance = np.linalg.norm(radial)
    normal = radial / distance
    tangent = np.asarray((-normal[1], normal[0]))
    s = radius * math.atan2(normal[1], normal[0])
    n = distance - radius
    return s, n, normal, tangent


def recover(nodes, elements, edges, stress, *, node=0, center=(0.0, 0.0), radius=1.0):
    return recover_fixed_physical_arc_patch_v3(
        nodes=nodes,
        elements=elements,
        stress_Pa=stress,
        boundary_node=node,
        cavity_id="owned-cavity",
        cavity_center_m=center,
        cavity_radius_m=radius,
        process_length_m=radius,
        poisson_ratio=0.3,
        owned_boundary_edges=edges,
        material_fingerprints=IDENTITY,
        state_fingerprint="accepted-state-sha",
    )


def test_constant_curvilinear_stress_oracle_and_frozen_contract():
    nodes, elements, edges = annular_mesh()

    def field(point):
        _, _, _, tangent = local_components(point)
        return 7.5e8 * np.outer(tangent, tangent)

    record = recover(nodes, elements, edges, stress_rows(nodes, elements, field))
    assert record["recovery_operator"] == RECOVERY_ID
    assert record["sample_count"] >= 12
    assert record["condition"]["maximum"] <= MAXIMUM_DESIGN_CONDITION
    assert record["traction_residual"] <= 1.0e-15
    assert record["plane_strain_sigma_zz_Pa"] == pytest.approx(0.3 * 7.5e8)
    assert record["mean_stress_Pa"] == pytest.approx(1.3 * 7.5e8 / 3.0)
    np.testing.assert_allclose(record["tensor_Pa"], ((0.0, 0.0), (0.0, 7.5e8)), rtol=1e-12)


def test_affine_curvilinear_stress_oracle_recovers_boundary_limit():
    nodes, elements, edges = annular_mesh()

    def field(point):
        s, n, normal, tangent = local_components(point)
        nn = 2.0e8 * n
        nt = 1.0e8 * n
        tt = 6.0e8 + 3.0e8 * n + 2.0e8 * s
        return (
            nn * np.outer(normal, normal)
            + tt * np.outer(tangent, tangent)
            + nt * (np.outer(normal, tangent) + np.outer(tangent, normal))
        )

    record = recover(nodes, elements, edges, stress_rows(nodes, elements, field))
    assert record["constrained_fit_residual"] <= 1.0e-12
    np.testing.assert_allclose(record["tensor_Pa"], ((0.0, 0.0), (0.0, 6.0e8)), rtol=1e-12)


def test_rigid_rotation_covariance_and_reversed_edge_ordering():
    nodes, elements, edges = annular_mesh()

    def field(point):
        _, _, _, tangent = local_components(point)
        return 4.0e8 * np.outer(tangent, tangent)

    stress = stress_rows(nodes, elements, field)
    base = recover(nodes, elements, edges, stress)
    angle = 0.37
    rotation = np.asarray(
        ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle)))
    )
    rotated_nodes = nodes @ rotation.T
    tensors = np.asarray(
        [
            rotation @ np.asarray(((xx, xy), (xy, yy))) @ rotation.T
            for xx, yy, xy in stress
        ]
    )
    rotated_stress = np.column_stack(
        (tensors[:, 0, 0], tensors[:, 1, 1], tensors[:, 0, 1])
    )
    rotated = recover(
        rotated_nodes,
        elements,
        list(reversed([(b, a) for a, b in edges])),
        rotated_stress,
    )
    np.testing.assert_allclose(
        rotated["tensor_Pa"],
        rotation @ np.asarray(base["tensor_Pa"]) @ rotation.T,
        rtol=1e-12,
        atol=1e-6,
    )
    assert (
        rotated["physical_arc_and_window_identity"]["sha256"]
        != base["physical_arc_and_window_identity"]["sha256"]
    )


def kirsch_tensor(point, *, radius=1.0, remote=1.0e9):
    x, y = point
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


def test_kirsch_family_has_two_quality_valid_fine_levels_and_fixed_window():
    rows = []
    for sectors in (64, 128, 256):
        nodes, elements, edges = annular_mesh(
            sectors=sectors, radial_layers=4, growth=2.0 * np.pi / sectors
        )
        node = sectors // 4
        record = recover(
            nodes, elements, edges, stress_rows(nodes, elements, kirsch_tensor), node=node
        )
        expected = np.asarray(((3.0e9, 0.0), (0.0, 0.0)))
        error = np.linalg.norm(np.asarray(record["tensor_Pa"]) - expected) / np.linalg.norm(expected)
        rows.append((record, error))
    for record, error in rows[-2:]:
        assert record["minimum_element_quality"] >= 0.05
        assert record["condition"]["maximum"] <= MAXIMUM_DESIGN_CONDITION
        assert error <= TENSOR_RELATIVE_TOLERANCE
        assert record["traction_residual"] <= TRACTION_RESIDUAL_TOLERANCE
        assert record["physical_arc_and_window_identity"]["boundary_position_m"] == pytest.approx((0.0, 1.0), abs=2e-15)
    assert rows[-1][1] <= rows[-2][1]
    assert len({row[0]["physical_arc_and_window_identity"]["sha256"] for row in rows}) == 1


def test_reflected_kirsch_pair_recovers_reflected_tensor():
    nodes, elements, edges = annular_mesh(sectors=64, radial_layers=4, growth=2.0 * np.pi / 64)
    stress = stress_rows(nodes, elements, kirsch_tensor)
    top = recover(nodes, elements, edges, stress, node=16)
    reflection = np.asarray(((1.0, 0.0), (0.0, -1.0)))
    mirrored_nodes = nodes @ reflection.T
    tensors = np.asarray(
        [reflection @ np.asarray(((xx, xy), (xy, yy))) @ reflection.T for xx, yy, xy in stress]
    )
    mirrored_stress = np.column_stack(
        (tensors[:, 0, 0], tensors[:, 1, 1], tensors[:, 0, 1])
    )
    bottom = recover(mirrored_nodes, elements[:, [0, 2, 1]], edges, mirrored_stress, node=16)
    np.testing.assert_allclose(
        bottom["tensor_Pa"],
        reflection @ np.asarray(top["tensor_Pa"]) @ reflection.T,
        rtol=1e-12,
        atol=1e-5,
    )


def test_inadequate_support_and_incomplete_ownership_fail_closed():
    nodes, elements, edges = annular_mesh(sectors=8, radial_layers=1)
    stress = stress_rows(nodes, elements, lambda _: np.eye(2))
    with pytest.raises(CavitySourceRecoveryV3Unavailable, match="inadequate solid support"):
        recover(nodes, elements, edges, stress)
    with pytest.raises(CavitySourceRecoveryV3Unavailable, match="incomplete material/state ownership"):
        recover_fixed_physical_arc_patch_v3(
            nodes=nodes,
            elements=elements,
            stress_Pa=stress,
            boundary_node=0,
            cavity_id="owned-cavity",
            cavity_center_m=(0.0, 0.0),
            cavity_radius_m=1.0,
            process_length_m=1.0,
            poisson_ratio=0.3,
            owned_boundary_edges=edges,
            material_fingerprints={},
            state_fingerprint="",
        )


def test_single_central_dbtt_v3_readiness_fails_closed_on_source_geometry_identity():
    import json
    from pathlib import Path
    from scripts.attest_cavity_source_recovery_v3_dbtt import OUTPUT, build_record

    retained = json.loads((Path(__file__).resolve().parents[1] / OUTPUT).read_text())
    observed = build_record()
    assert observed["DBTT_SOURCE_READINESS"] == "BLOCKED_WITH_EXACT_V3_FAILURE_CLASS"
    assert observed["exact_v3_failure_class"] == ["SOURCE_GEOMETRY_IDENTITY"]
    assert observed["scientific_unavailability"] == (
        "the exact source coordinate is inconsistent with the owned cavity"
    )
    assert observed["requested_refinement_levels"] == 3
    assert observed["refinement_levels_run"] == 0
    assert observed["accepted_pre_source_state_unchanged_on_noncertification"] is True
    assert observed["thresholds_and_rng_unchanged_on_noncertification"] is True
    assert observed["unexpected_programming_exceptions_caught"] is False
    assert observed["oracle_states_accepted"] == 0
    assert observed["paired_trajectories_run"] == 0
    assert observed["next_bounded_step"] == "DERIVE_FINITE_ACTIVATION_ZONE_WORK_OBSERVABLE"
    assert observed["source_geometry_diagnostic"] == retained["source_geometry_diagnostic"]
    assert observed["ligament_energy_gate"] == retained["ligament_energy_gate"]
