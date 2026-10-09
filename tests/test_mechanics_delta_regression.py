from dataclasses import replace

import numpy as np
import pytest

from arrhenius_fracture.coalescence import segment_intersection_first
from arrhenius_fracture.config import ElasticProperties
from arrhenius_fracture.fem import assemble_mechanics, plane_strain_D, stress_state
from arrhenius_fracture.mesh import rebuild_tri_mesh, restore_tri_mesh


def simple_mesh():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    elems = np.array([[0, 1, 2]], dtype=np.int32)
    return rebuild_tri_mesh(nodes, elems, tip_centers=[[0.0, 0.0]])


@pytest.mark.parametrize("a,b,expected", [
    ((2.0, 0.0), (3.0, 0.0), None),
    ((1.0, 0.0), (2.0, 0.0), 1.0),
    ((0.25, 0.0), (0.75, 0.0), 0.25),
    ((0.75, 0.0), (0.25, 0.0), 0.25),
    ((-1.0, 0.0), (2.0, 0.0), 1.0e-8),
])
def test_collinear_overlap_first_intersection_cases(a, b, expected):
    result = segment_intersection_first(np.array([0.0, 0.0]), np.array([1.0, 0.0]),
                                        np.array(a), np.array(b))
    if expected is None:
        assert result is None
    else:
        assert result[0] == pytest.approx(expected)


def test_noncollinear_crossing_and_parallel_disjoint():
    crossing = segment_intersection_first(np.array([0.0, 0.0]), np.array([1.0, 0.0]),
                                          np.array([0.5, -1.0]), np.array([0.5, 1.0]))
    assert crossing[0] == pytest.approx(0.5)
    assert segment_intersection_first(np.array([0.0, 0.0]), np.array([1.0, 0.0]),
                                      np.array([0.0, 1.0]), np.array([1.0, 1.0])) is None


def test_element_damage_is_authoritative_and_nodal_fallback_is_preserved():
    mesh = simple_mesh()
    mat = ElasticProperties(E=200e9, nu=0.3)
    D = plane_strain_D(mat)
    u = np.array([0.0, 0.0, 1e-3, 0.0, 0.0, 0.0])
    ep = np.zeros((3, 1)); rho = np.zeros(1)
    nodal = np.array([0.0, 0.0, 0.0])
    base = assemble_mechanics(mesh, u, ep, rho, nodal, D, mat)
    p0_mesh = replace(mesh, element_damage_gp=np.array([0.5]))
    p0 = assemble_mechanics(p0_mesh, u, ep, rho, nodal, D, mat)
    assert np.linalg.norm(p0[2]) < np.linalg.norm(base[2])
    assert np.linalg.norm(p0[2]) / np.linalg.norm(base[2]) == pytest.approx(
        ((1.0 - 0.5) ** 2 + 1e-6) / (1.0 + 1e-6), rel=1e-10)
    base_stress = stress_state(mesh, u, ep, nodal, D, mat)[0]
    p0_stress = stress_state(p0_mesh, u, ep, nodal, D, mat)[0]
    np.testing.assert_allclose(base_stress, base[2])
    np.testing.assert_allclose(p0_stress, p0[2])


def test_restore_accepts_ulp_drift_preserves_authoritative_value_and_rejects_geometry_change():
    mesh = simple_mesh()
    expected_h = np.nextafter(mesh.hbar, np.inf)
    expected_tip = np.nextafter(mesh.hbar_tip, np.inf)
    metadata = {"hbar_m": expected_h, "hbar_tip_m": expected_tip,
                "tip_reference_centers_m": [[0.0, 0.0]]}
    restored = restore_tri_mesh(mesh.nodes, mesh.elems.astype(np.int32), metadata)
    assert restored.hbar == expected_h
    assert restored.hbar_tip == expected_tip
    assert restored.elems.dtype == np.int32
    changed = dict(metadata, hbar_m=mesh.hbar * (1.0 + 1e-7))
    with pytest.raises(ValueError, match="recomputation tolerance"):
        restore_tri_mesh(mesh.nodes, mesh.elems, changed)
