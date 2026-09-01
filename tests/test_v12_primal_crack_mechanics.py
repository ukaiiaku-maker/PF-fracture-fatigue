import numpy as np

from arrhenius_fracture.conforming_crack_oracle_v12 import (
    CONFORMING_ORACLE_SOURCE_COMMIT, build_matched_crack_parent, conforming_slit_from_parent,
)
from arrhenius_fracture.primal_crack_mechanics_v12 import run_straight_case


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
