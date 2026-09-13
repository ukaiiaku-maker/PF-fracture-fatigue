import numpy as np
import pytest

from arrhenius_fracture.arc_resolved_cavity_traction_v1 import (
    diagnose_arc_resolved_cavity_traction_v1,
)


def _four_triangle_ring():
    nodes = np.asarray(((1., 0.), (0., 1.), (-1., 0.), (0., -1.),
                        (2., 0.), (0., 2.), (-2., 0.), (0., -2.)))
    elements = np.asarray(((0, 4, 1), (1, 5, 2), (2, 6, 3), (3, 7, 0)))
    edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    return nodes, elements, edges


def _diagnosis(edges=None):
    nodes, elements, original = _four_triangle_ring()
    stress = np.tile(np.asarray((3., 7., 2.))[:, None], (1, len(elements)))
    return diagnose_arc_resolved_cavity_traction_v1(
        nodes=nodes, elements=elements, stress_Pa=stress,
        owned_boundary_edges=original if edges is None else edges,
        cavity_center_m=(0., 0.), cavity_radius_m=1.,
        source_position_m=(1., 0.), junction_position_m=(-1., 0.),
        process_length_m=1., remote_traction_scale_Pa=10.,
        plastic_strain=np.zeros((3, len(elements))), density_m2=np.arange(len(elements)),
    )


def test_arc_rows_recompute_scalar_and_partition_complete_boundary():
    result = _diagnosis()
    rows = result["edge_rows"]
    expected = np.sqrt(sum(row["traction_l2_numerator_Pa2_m"] for row in rows)
                       / sum(row["edge_length_m"] for row in rows)) / 10.
    assert result["recomputed_raw_adjacent_element_traction_normalized"] == pytest.approx(expected)
    assert result["edge_count"] == 4
    assert result["region_summaries"]["complete_cavity_boundary"]["edge_count"] == 4


def test_edge_order_and_orientation_do_not_change_diagnosis():
    _, _, edges = _four_triangle_ring()
    forward = _diagnosis(edges)
    reverse = _diagnosis([edge[::-1] for edge in edges[::-1]])
    assert forward["edge_rows"] == reverse["edge_rows"]
    assert forward["region_summaries"] == reverse["region_summaries"]


def test_normals_membership_weighting_and_unique_ownership_are_explicit():
    result = _diagnosis()
    for row in result["edge_rows"]:
        assert row["polygon_nominal_normal_dot"] > 0.
        assert row["traction_l2_numerator_Pa2_m"] == pytest.approx(
            np.dot(row["traction_vector_Pa"], row["traction_vector_Pa"])
            * row["edge_length_m"]
        )
    assert result["region_summaries"]["fixed_source_window"]["edge_count"] == sum(
        abs(row["arc_coordinate_from_source_m"]) <= result["source_window_half_width_m"]
        for row in result["edge_rows"]
    )
    _, _, edges = _four_triangle_ring()
    with pytest.raises(ValueError, match="duplicate"):
        _diagnosis(list(edges) + [edges[0]])
    with pytest.raises(ValueError, match="exactly one"):
        _diagnosis([(0, 5), *edges])
