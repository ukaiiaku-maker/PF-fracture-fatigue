import numpy as np

from arrhenius_fracture.conforming_crack_oracle_v12 import (
    CONFORMING_ORACLE_SOURCE_COMMIT, build_matched_crack_parent, conforming_slit_from_parent,
)
from types import SimpleNamespace
from arrhenius_fracture.fem import plane_strain_D
from arrhenius_fracture.config import ElasticProperties
from arrhenius_fracture.mesh import rebuild_tri_mesh
from arrhenius_fracture.primal_crack_mechanics_v12 import _interface_tractions, _stress_tensor, _traction_local_components, recover_element_fields, run_rotated_cases, run_straight_case


def test_affine_patch_recovers_interleaved_strain_stress_and_energy_on_rigidly_moved_triangles():
    coefficients=np.array((.17,-.23,.41,.31,-.19,-.37))
    expected_strain=np.array((coefficients[0],coefficients[4],coefficients[1]+coefficients[3]))
    constitutive=plane_strain_D(ElasticProperties(E=73e9,nu=.27))
    base=np.array(((.1,-.2),(1.4,.3),(.35,1.2)))
    for angle,shift in ((0.,(0.,0.)),(.63,(2.7,-1.1))):
        rotation=np.array(((np.cos(angle),-np.sin(angle)),(np.sin(angle),np.cos(angle))))
        nodes=base@rotation.T+shift
        mesh=rebuild_tri_mesh(nodes,np.array(((0,1,2),)))
        x,y=nodes.T
        ux=coefficients[0]*x+coefficients[1]*y+coefficients[2]
        uy=coefficients[3]*x+coefficients[4]*y+coefficients[5]
        u=np.column_stack((ux,uy)).ravel()
        strain,stress=recover_element_fields(mesh,u,constitutive)
        np.testing.assert_allclose(strain[:,0],expected_strain,rtol=0,atol=2e-15)
        np.testing.assert_allclose(stress[:,0],constitutive@expected_strain,rtol=2e-15,atol=1e-5)
        analytical=.5*mesh.area_e[0]*expected_strain@(constitutive@expected_strain)
        element=.5*mesh.area_e[0]*strain[:,0]@stress[:,0]
        stiffness=mesh.area_e[0]*mesh.B_e[0].T@constitutive@mesh.B_e[0]
        quadratic=.5*u@stiffness@u
        np.testing.assert_allclose((element,quadratic),(analytical,analytical),rtol=5e-15)


def test_affine_patch_detects_superseded_blocked_dof_recovery_order():
    mesh=rebuild_tri_mesh(np.array(((.2,-.1),(1.1,.4),(.3,1.3))),np.array(((0,1,2),)))
    x,y=mesh.nodes.T; u=np.column_stack((.17*x-.23*y+.41,.31*x-.19*y-.37)).ravel()
    correct,_=recover_element_fields(mesh,u)
    blocked=np.r_[u[0::2],u[1::2]]
    wrong=np.einsum("eij,ej->ei",mesh.B_e,blocked[None,:]).T
    assert np.linalg.norm(wrong-correct)>1e-2


def test_conforming_oracle_records_pr57_source_and_normalizes_to_parent():
    parent=build_matched_crack_parent(8e-4,8e-4,(2e-4,0.),(5e-4,0.),25e-6)
    slit=conforming_slit_from_parent(parent)
    assert CONFORMING_ORACLE_SOURCE_COMMIT=="8ad7f42c49c27d1066ff5ef7ee4a910232f2e7d4"
    np.testing.assert_array_equal(slit.parent_node_of_node[slit.mesh.elems],parent.mesh.elems)
    assert slit.mesh.ne==parent.mesh.ne and slit.mesh.nn>parent.mesh.nn


def test_coarse_straight_screen_has_all_four_representations_and_equilibrates():
    rows,derivatives=run_straight_case(h_values=(25e-6,),kappas=(1e-6,))
    assert {r["representation"] for r in rows}=={"A_INTACT","B_V11","C_V12","D_CONFORMING"}
    assert max(r["free_residual_relative"] for r in rows)<1e-10
    assert max(r["energy_reaction_identity_relative"] for r in rows)<1e-10
    assert {r["representation"] for r in derivatives}=={"V12","CONF"}
    conforming=next(r for r in rows if r["representation"]=="D_CONFORMING")
    v12=next(r for r in rows if r["representation"]=="C_V12")
    assert conforming["conforming_extrapolated_direct_cod_error"]<.05
    assert v12["crack_opening_reference_error"]>.05  # required failure cannot be hidden by global PASS
    assert max(v12[k] for k in ("pin_reaction_relative_error","pin_energy_relative_error","pin_cod_relative_error"))<1e-10
    assert max(r[k] for r in rows for k in r if k.startswith("mirror_") and k.endswith("relative"))<.05


def test_coarse_rotated_screen_preserves_matched_representations():
    rows=run_rotated_cases(angles=(30.,),h_values=(25e-6,),kappas=(1e-6,))
    assert {r["representation"] for r in rows}=={"A_INTACT","B_V11","C_V12","D_CONFORMING"}
    assert max(r["free_residual_relative"] for r in rows)<1e-10


def test_interface_traction_jump_is_zero_for_known_uniform_field():
    parent=build_matched_crack_parent(8e-4,8e-4,(2e-4,0.),(5e-4,0.),25e-6)
    cent=parent.mesh.nodes[parent.mesh.elems].mean(axis=1); mask=(cent[:,0]>=2e-4)&(cent[:,0]<=5e-4)&(np.abs(cent[:,1])<25e-6)
    result=SimpleNamespace(sigma_gp=np.tile(np.array([[2.],[3.],[.5]]),(1,parent.mesh.ne)))
    measured=_interface_tractions(parent.mesh,result,mask,parent.p0,parent.p1)
    assert measured["trimmed_interior_jump_traction_rms_Pa"]<1e-12


def test_affine_stress_traction_on_arbitrary_cut_and_normal_reversal():
    stress=np.array((7.0,3.0,1.5)); interface_normal=np.array((2.0,-1.0)); interface_normal/=np.linalg.norm(interface_normal); crack_normal=np.array((.6,.8)); tangent=np.array((.8,-.6))
    expected=_stress_tensor(stress)@interface_normal
    measured=_traction_local_components(stress,interface_normal,crack_normal,tangent)
    np.testing.assert_allclose(measured,(expected@crack_normal,expected@tangent),rtol=0,atol=1e-14)
    reversed_components=_traction_local_components(stress,-interface_normal,crack_normal,tangent)
    np.testing.assert_allclose(reversed_components,-measured,rtol=0,atol=1e-14)
    np.testing.assert_allclose(np.abs(reversed_components),np.abs(measured),rtol=0,atol=0)


def test_conforming_slit_has_no_connectivity_transmitting_across_open_faces():
    parent=build_matched_crack_parent(8e-4,8e-4,(2e-4,0.),(5e-4,0.),25e-6); slit=conforming_slit_from_parent(parent)
    upper=set(slit.upper_face_edges[1:-1].ravel()); lower=set(slit.lower_face_edges[1:-1].ravel())
    assert upper.isdisjoint(lower)
    assert not any(any(node in upper for node in elem) and any(node in lower for node in elem) for elem in slit.mesh.elems)


def test_mode_i_interface_regions_balance_and_shear_is_numerical_zero():
    rows,_=run_straight_case(h_values=(25e-6,),kappas=(1e-6,)); row=next(r for r in rows if r["representation"]=="C_V12")
    assert all(row[f"{region}_interface_length_m"]>0 for region in ("root","trimmed_interior","active_tip"))
    assert row["discrete_upper_lower_normal_balance_relative"]<1e-10
    assert abs(row["discrete_signed_transmitted_shear_force_N_per_m"])/abs(row["reaction_N_per_m"])<1e-10
    assert abs(abs(row["discrete_upper_shear_force_N_per_m"])-abs(row["discrete_lower_shear_force_N_per_m"]))/abs(row["reaction_N_per_m"])<1e-10
