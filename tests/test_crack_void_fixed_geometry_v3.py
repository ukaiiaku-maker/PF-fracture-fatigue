import numpy as np
import pytest

from arrhenius_fracture.crack_void_mechanics_v5 import solve_crack_void_case
from arrhenius_fracture.explicit_cavity_v5 import build_explicit_hole_mesh, conform_crack_path


MODE = "V3_FIXED_LABORATORY_GEOMETRY"
PATH = ((0.0, 0.0), (5.0e-4, 0.0))


def solve(**kwargs):
    return solve_crack_void_case(crack_path_m=PATH, geometry_mode=MODE, **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"cavity_center_m": (7.2e-4, 0.0)},
    {"cavity_radius_m": 4.0e-5},
    {"boundary_segments": 40, "radial_layers": 16},
    {"cavity_center_m": (7.0e-4, 4.0e-5)},
    {"cavity_center_m": (7.0e-4, -4.0e-5)},
])
def test_fixed_laboratory_crack_is_invariant_to_cavity_and_mesh(kwargs):
    result = solve(**kwargs)
    observed = result["observables"]
    assert np.allclose(observed["crack_root_m"], PATH[0], rtol=0.0, atol=1e-12)
    assert np.allclose(observed["crack_tip_m"], PATH[-1], rtol=0.0, atol=1e-12)
    assert result["geometry_conformity_audit"]["maximum_requested_realized_error_m"] <= 1e-12


def test_polygon_intersection_not_ideal_circle_is_the_physical_ligament():
    result = solve(boundary_segments=32)
    observed = result["observables"]
    assert observed["ray_intersects_polygon"]
    assert observed["ligament_length_m"] == pytest.approx(
        observed["realized_polygon_intersection_m"][0] - PATH[-1][0])
    assert observed["polygonization_intersection_error_m"] > 0.0
    assert "radius_over_nominal_h" not in observed
    assert observed["R_over_h_cavity_max"] > 0.0


def test_offsets_are_crack_relative_and_share_identical_crack():
    positive = solve(cavity_center_m=(7.0e-4, 4.0e-5))["observables"]
    negative = solve(cavity_center_m=(7.0e-4, -4.0e-5))["observables"]
    assert positive["crack_root_m"] == negative["crack_root_m"]
    assert positive["crack_tip_m"] == negative["crack_tip_m"]
    assert positive["signed_normal_offset_over_R"] == pytest.approx(
        -negative["signed_normal_offset_over_R"])


def test_complete_kinked_path_is_realized_and_sets_tip_tangent():
    path = ((0.0, 0.0), (2.5e-4, 0.0), (5.0e-4, 5.0e-5))
    hole = build_explicit_hole_mesh(1e-3, 1e-3, (7e-4, 3e-5), 5e-5, 5e-5, 32,
                                    radial_layers_override=12)
    realized, audit = conform_crack_path(hole, path)
    records = audit["path_records"]
    assert np.allclose([row["realized_m"] for row in records], path, rtol=0.0, atol=1e-12)
    assert realized.validation["geometry_generation"] >= 2


def test_v3_fails_closed_without_path_and_outside_points():
    with pytest.raises(ValueError, match="requires explicit crack_path_m"):
        solve_crack_void_case(geometry_mode=MODE)
    with pytest.raises(ValueError, match="outside the specimen mesh"):
        solve_crack_void_case(crack_path_m=((0.0, 0.0), (2.0e-3, 0.0)), geometry_mode=MODE)
    with pytest.raises(ValueError, match="actual exterior boundary"):
        solve_crack_void_case(crack_path_m=((1.0e-4, 0.0), (5.0e-4, 0.0)), geometry_mode=MODE)


def test_crack_and_cavity_derivatives_change_only_the_declared_geometry():
    crack_a = solve_crack_void_case(crack_path_m=PATH, geometry_mode=MODE)
    crack_b = solve_crack_void_case(crack_path_m=(PATH[0], (5.1e-4, 0.0)), geometry_mode=MODE)
    assert crack_a["observables"]["cavity_center_m"] == crack_b["observables"]["cavity_center_m"]
    cavity_b = solve(cavity_radius_m=5.1e-5)
    assert cavity_b["observables"]["crack_root_m"] == crack_a["observables"]["crack_root_m"]
    assert cavity_b["observables"]["crack_tip_m"] == crack_a["observables"]["crack_tip_m"]


def test_cavity_traction_diagnostic_separates_weak_and_recovered_quantities():
    coarse = solve_crack_void_case(crack_enabled=False, boundary_segments=32, radial_layers=12)["observables"]
    fine = solve_crack_void_case(crack_enabled=False, boundary_segments=64, radial_layers=24)["observables"]
    assert coarse["weak_cavity_boundary_residual_relative"] < 1e-10
    assert fine["weak_cavity_boundary_residual_relative"] < 1e-10
    assert fine["cavity_traction_l2_normalized"] < coarse["cavity_traction_l2_normalized"]
    assert fine["cavity_traction_normal_l2_normalized"] > 0.0
    assert fine["cavity_traction_tangential_l2_normalized"] > 0.0
    assert len(fine["cavity_traction_resultant_normalized"]) == 2
    assert abs(fine["cavity_traction_moment_normalized"]) < 1e-10
