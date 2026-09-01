import numpy as np

from arrhenius_fracture.conforming_crack_oracle_v12 import (
    CONFORMING_ORACLE_SOURCE_COMMIT, build_matched_crack_parent, conforming_slit_from_parent,
)
from types import SimpleNamespace
from arrhenius_fracture.primal_crack_mechanics_v12 import _interface_tractions, run_rotated_cases, run_straight_case


def test_conforming_oracle_records_pr57_source_and_normalizes_to_parent():
    parent=build_matched_crack_parent(8e-4,8e-4,(2e-4,0.),(5e-4,0.),25e-6)
    slit=conforming_slit_from_parent(parent)
    assert CONFORMING_ORACLE_SOURCE_COMMIT=="8ad7f42"
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
