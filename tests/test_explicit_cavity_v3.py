import numpy as np
import pytest
from arrhenius_fracture.explicit_cavity_v3 import build_explicit_hole_mesh,fill_explicit_hole_mesh,solve_static_hole,triangle_intersects_open_disk

def test_disk_intersection_catches_edge_and_center():
 tri=np.array(((0.,0.),(2.,0.),(0.,2.)))
 assert triangle_intersects_open_disk(tri,(.5,.5),.1)
 assert triangle_intersects_open_disk(tri,(1.,-.05),.1)

@pytest.mark.parametrize("h,n",((50e-6,32),(25e-6,64)))
def test_actual_closed_traction_free_cavity_geometry(h,n):
 hole=build_explicit_hole_mesh(1e-3,1e-3,(5e-4,0.),1e-4,h,n); v=hole.validation
 assert v["actual_internal_components"]==1 and v["cavity_cycle"]
 assert v["triangle_disk_intersections"]==0 and v["orphan_nodes"]==0
 assert v["polygon_exact_node_set_match"] and len(np.unique(hole.mesh.elems))==hole.mesh.nn

def test_static_cavity_equilibrates_with_finite_mechanics_and_free_boundary():
 hole=build_explicit_hole_mesh(1e-3,1e-3,(5e-4,0.),1e-4,35e-6,64)
 result=solve_static_hole(hole,1e-7)
 assert np.isfinite(result.reaction_top_N_per_m) and np.isfinite(result.stored_energy_J_per_m)
 assert result.free_residual_norm_N_per_m<1e-6*abs(result.reaction_top_N_per_m)
 assert result.traction_l2_normalized<.15 and result.symmetry_error<.1

def test_filled_control_changes_only_cavity_patch():
 hole=build_explicit_hole_mesh(1e-3,1e-3,(5e-4,0.),1e-4,50e-6,32); filled=fill_explicit_hole_mesh(hole)
 np.testing.assert_array_equal(filled.mesh.elems[:hole.mesh.ne],hole.mesh.elems)
 np.testing.assert_array_equal(filled.mesh.nodes[:hole.mesh.nn],hole.mesh.nodes)
