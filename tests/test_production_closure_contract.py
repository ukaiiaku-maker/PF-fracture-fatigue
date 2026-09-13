import numpy as np
import pytest

from arrhenius_fracture.equilibrated_boundary_traction_recovery_v1 import (
    recover_equilibrated_boundary_traction_v1,
)
from arrhenius_fracture.production_closure_contract import (
    ACCEPTANCE_LIMITS, CONTRACT_ID, FALLBACK_OPERATOR_ID, contract_record,
)


def _annulus(radial_layers=5, sectors=48, rotation=0.0, reflected=False):
    theta = rotation + np.arange(sectors) * 2 * np.pi / sectors
    if reflected:
        theta = -theta
    radii = np.linspace(1.0, 1.5, radial_layers + 1)
    nodes = np.vstack([
        np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
        for radius in radii
    ])
    elements = []
    for layer in range(radial_layers):
        inner = layer * sectors
        outer = (layer + 1) * sectors
        for index in range(sectors):
            following = (index + 1) % sectors
            pair = ((inner + index, outer + index, outer + following),
                    (inner + index, outer + following, inner + following))
            for triangle in pair:
                xy = nodes[list(triangle)]
                first, second = xy[1] - xy[0], xy[2] - xy[0]
                if first[0] * second[1] - first[1] * second[0] < 0:
                    triangle = (triangle[0], triangle[2], triangle[1])
                elements.append(triangle)
    edges = [(index, (index + 1) % sectors) for index in range(sectors)]
    source = int(np.argmax(nodes[:sectors, 0]))
    return nodes, np.asarray(elements), edges, source


def _run(stress_function, **mesh_kwargs):
    nodes, elements, edges, source = _annulus(**mesh_kwargs)
    centroids = nodes[elements].mean(axis=1)
    tensor = stress_function(centroids)
    return recover_equilibrated_boundary_traction_v1(
        nodes=nodes, elements=elements, stress_Pa=tensor,
        boundary_node=source, cavity_center_m=(0.0, 0.0), cavity_radius_m=1.0,
        process_length_m=1.0, owned_boundary_edges=edges,
        remote_traction_scale_Pa=10.0, state_fingerprint="manufactured",
    )


def _constant(points):
    tensor = np.zeros((len(points), 2, 2))
    tensor[:, 0, 0] = 3.0
    tensor[:, 1, 1] = 7.0
    tensor[:, 0, 1] = tensor[:, 1, 0] = 2.0
    return tensor


def _affine(points):
    x, y = points.T
    tensor = np.zeros((len(points), 2, 2))
    tensor[:, 0, 0] = 3.0 + 2.0 * x + 0.5 * y
    tensor[:, 1, 1] = 7.0 - 1.5 * x + 4.0 * y
    tensor[:, 0, 1] = tensor[:, 1, 0] = 2.0 - 4.0 * x - 2.0 * y
    return tensor


def _kirsch(points):
    x, y = points.T
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    c2, s2 = np.cos(2 * theta), np.sin(2 * theta)
    rr = 5 * (1 - r**-2) + 5 * (1 - 4*r**-2 + 3*r**-4) * c2
    tt = 5 * (1 + r**-2) - 5 * (1 + 3*r**-4) * c2
    rt = -5 * (1 + 2*r**-2 - 3*r**-4) * s2
    er = np.column_stack((np.cos(theta), np.sin(theta)))
    et = np.column_stack((-np.sin(theta), np.cos(theta)))
    return (rr[:, None, None] * er[:, :, None] * er[:, None, :]
            + tt[:, None, None] * et[:, :, None] * et[:, None, :]
            + rt[:, None, None] * (er[:, :, None] * et[:, None, :]
                                    + et[:, :, None] * er[:, None, :]))


def test_contract_freezes_final_levels_limits_and_independent_fallback():
    record = contract_record()
    assert record["contract"] == CONTRACT_ID
    assert [(row["name"], row["first_strip_radial_subdivisions"])
            for row in record["final_local_levels"]] == [("D", 16), ("E", 32)]
    assert ACCEPTANCE_LIMITS["raw_adjacent_cst_traction"] == 0.05
    assert ACCEPTANCE_LIMITS["equilibrated_boundary_traction"] == 0.05
    assert record["fallback"]["operator"] == FALLBACK_OPERATOR_ID
    assert record["fallback"]["zero_boundary_traction_imposed"] is False
    assert record["immutable_model_boundary"]["r_tip_law_changed"] is False
    assert record["immutable_model_boundary"]["r_tip_equals_R_void"] is False


@pytest.mark.parametrize("field", (_constant, _affine))
def test_manufactured_equilibrated_fields_are_recovered(field):
    result = _run(field)
    expected = field(np.asarray(((1.0, 0.0),)))[0]
    assert np.asarray(result["source_tensor_Pa"]) == pytest.approx(expected, abs=2e-10)
    assert result["fit_residual"] < 1e-11
    assert result["zero_boundary_traction_imposed"] is False


def test_kirsch_rotation_reflection_edge_order_and_two_levels():
    coarse = _run(_kirsch, radial_layers=5, sectors=48)
    fine = _run(_kirsch, radial_layers=8, sectors=96)
    rotated = _run(_kirsch, radial_layers=8, sectors=96, rotation=np.pi / 7)
    reflected = _run(_kirsch, radial_layers=8, sectors=96, reflected=True)
    assert coarse["normalized_boundary_traction"] <= 0.05
    assert fine["normalized_boundary_traction"] <= 0.05
    assert abs(fine["normalized_boundary_traction"] - coarse["normalized_boundary_traction"]) <= 0.05
    assert rotated["normalized_boundary_traction"] <= 0.05
    assert reflected["normalized_boundary_traction"] <= 0.05


def test_edge_and_normal_orientation_do_not_author_traction():
    nodes, elements, edges, source = _annulus(radial_layers=8, sectors=96)
    stress = _kirsch(nodes[elements].mean(axis=1))
    kwargs = dict(
        nodes=nodes, elements=elements, stress_Pa=stress, boundary_node=source,
        cavity_center_m=(0.0, 0.0), cavity_radius_m=1.0, process_length_m=1.0,
        remote_traction_scale_Pa=10.0, state_fingerprint="orientation",
    )
    forward = recover_equilibrated_boundary_traction_v1(owned_boundary_edges=edges, **kwargs)
    reverse = recover_equilibrated_boundary_traction_v1(
        owned_boundary_edges=[edge[::-1] for edge in edges[::-1]], **kwargs,
    )
    assert forward["normalized_boundary_traction"] == pytest.approx(
        reverse["normalized_boundary_traction"], rel=0, abs=1e-14,
    )
    assert forward["window_identity"] == reverse["window_identity"]


@pytest.mark.parametrize("level,radial", (("D", 16), ("E", 32)))
def test_final_geometry_levels_pass_construction_before_fem(level, radial):
    from arrhenius_fracture.cavity_source_shape_regular_mesh_v5 import (
        build_shape_regular_source_patch_hole_mesh,
        discrete_cavity_boundary_fingerprint,
    )

    hole = build_shape_regular_source_patch_hole_mesh(
        1.0e-3, 1.0e-3, (7.0e-4, 0.0), 5.5e-5, 5.0e-5,
        local_level=level, polygon_sectors=128,
    )
    edges = np.sort(np.concatenate((
        hole.mesh.elems[:, (0, 1)], hole.mesh.elems[:, (1, 2)],
        hole.mesh.elems[:, (2, 0)],
    )), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    owners = {tuple(edge): int(count) for edge, count in zip(unique, counts)}
    assert all(owners[tuple(sorted(map(int, edge)))] == 1 for edge in hole.cavity_edges)
    assert len(np.unique(hole.cavity_edges)) == len(hole.cavity_edges)
    assert np.all(hole.mesh.area_e > 0.0)
    assert set(np.unique(hole.mesh.elems)) == set(range(hole.mesh.nn))
    assert hole.validation["minimum_quality"] >= 0.10
    assert hole.validation["first_strip_radial_subdivisions"] == radial
    assert any(np.linalg.norm(point - np.asarray((7.55e-4, 0.0))) <= 1.0e-12
               for point in hole.mesh.nodes)
    assert discrete_cavity_boundary_fingerprint(hole) == (
        "b1dd9e1aa4952807ac081036fe5be5a579df9ccd297af2e7a4162374fcce8cb7"
    )


def test_longest_edge_refinement_preserves_frozen_boundary_edges():
    from arrhenius_fracture.adaptive_multitip_mesh_v11 import _subdivide

    nodes = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)))
    elements = np.asarray(((0, 1, 2), (1, 3, 2)))
    refined_nodes, refined_elements, midpoints, *_ = _subdivide(
        nodes, elements, (0, 1), longest_edge_closure=True,
        protected_edges=((0, 1),),
    )
    edge_counts = {}
    for triangle in refined_elements:
        for edge in ((triangle[0], triangle[1]), (triangle[1], triangle[2]),
                     (triangle[2], triangle[0])):
            key = tuple(sorted(map(int, edge)))
            edge_counts[key] = edge_counts.get(key, 0) + 1
    assert (0, 1) not in midpoints
    assert edge_counts[(0, 1)] == 1
    assert not any(np.array_equal(point, (0.5, 0.0)) for point in refined_nodes[4:])
