from types import SimpleNamespace

import numpy as np
import pytest

from arrhenius_fracture.unit_slip_perturbation_v1026 import (
    SlipRibbonPerturbation,
    slip_ribbon_eigenstrain_increment,
)


def _mesh():
    # Four small triangles centered along a horizontal slip ribbon.
    nodes = np.array(
        [
            [0.0, -0.1],
            [0.0, 0.1],
            [0.5, -0.1],
            [0.5, 0.1],
            [1.0, -0.1],
            [1.0, 0.1],
        ]
    )
    elems = np.array([[0, 2, 3], [0, 3, 1], [2, 4, 5], [2, 5, 3]])
    area = np.full(len(elems), 0.05)
    return SimpleNamespace(nodes=nodes, elems=elems, area_e=area, ne=len(elems))


def _perturbation(sign=1.0):
    return SlipRibbonPerturbation(
        system=0,
        region="active",
        bin_index=0,
        start_xy_m=np.array([0.0, 0.0]),
        end_xy_m=np.array([1.0, 0.0]),
        slip_direction=np.array([1.0, 0.0]),
        plane_normal=np.array([0.0, 1.0]),
        width_m=0.25,
        burgers_m=2.5e-10,
        signed_line_content=sign * 2.0,
    )


def test_slip_ribbon_normalization_and_burgers_sign_reversal():
    mesh = _mesh()
    positive, audit = slip_ribbon_eigenstrain_increment(mesh, _perturbation(+1.0))
    negative, _ = slip_ribbon_eigenstrain_increment(mesh, _perturbation(-1.0))
    expected_gamma = 2.0 * 2.5e-10 / 0.25
    assert _perturbation(+1.0).plastic_shear == pytest.approx(expected_gamma)
    assert np.max(np.abs(positive[2])) == pytest.approx(expected_gamma)
    assert negative == pytest.approx(-positive)
    assert audit["selected_elements"] == 4
    assert audit["normalization"] == "delta_N_line = gamma * width / b"
    assert audit["fitted_to_toughness_or_fatigue"] is False


def test_slip_ribbon_requires_mesh_support():
    mesh = _mesh()
    p = SlipRibbonPerturbation(
        system=0,
        region="active",
        bin_index=0,
        start_xy_m=np.array([0.0, 2.0]),
        end_xy_m=np.array([1.0, 2.0]),
        slip_direction=np.array([1.0, 0.0]),
        plane_normal=np.array([0.0, 1.0]),
        width_m=0.01,
        burgers_m=2.5e-10,
        signed_line_content=1.0,
    )
    with pytest.raises(ValueError, match="selects no FEM elements"):
        slip_ribbon_eigenstrain_increment(mesh, p)
