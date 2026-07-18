from types import SimpleNamespace

import numpy as np
import pytest

from arrhenius_fracture.config import ElasticProperties
from arrhenius_fracture.interaction_integral_v1026 import (
    _auxiliary_displacement_local,
    _auxiliary_stress_local,
    compute_signed_interaction_integral,
    isotropic_plane_strain_D,
    require_isotropic_stiffness,
)


def _mesh(n=31, radius=1.0):
    x = np.linspace(-radius, radius, n)
    y = np.linspace(-radius, radius, n)
    X, Y = np.meshgrid(x, y)
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    elems = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            b = a + 1
            c = a + n
            d = c + 1
            elems.extend(([a, b, d], [a, d, c]))
    elems = np.asarray(elems, dtype=int)
    dNdx = np.zeros((len(elems), 2, 3))
    area = np.zeros(len(elems))
    for index, conn in enumerate(elems):
        p = nodes[conn]
        interpolation = np.array(
            [
                [1.0, p[0, 0], p[0, 1]],
                [1.0, p[1, 0], p[1, 1]],
                [1.0, p[2, 0], p[2, 1]],
            ]
        )
        inverse = np.linalg.inv(interpolation)
        dNdx[index, 0] = inverse[1]
        dNdx[index, 1] = inverse[2]
        area[index] = 0.5 * abs(
            np.linalg.det(np.column_stack([p[1] - p[0], p[2] - p[0]]))
        )
    return SimpleNamespace(
        nodes=nodes,
        elems=elems,
        dNdx_e=dNdx,
        area_e=area,
        ne=len(elems),
    )


def _exact_state(mesh, mode, K, mat):
    nodal = np.zeros((len(mesh.nodes), 2))
    for index, (x, y) in enumerate(mesh.nodes):
        if np.hypot(x, y) > 1.0e-14:
            nodal[index] = _auxiliary_displacement_local(
                x, y, mode, mat.E, mat.nu, K
            )
    centroids = mesh.nodes[mesh.elems].mean(axis=1)
    sigma = np.zeros((3, mesh.ne))
    for element, (x, y) in enumerate(centroids):
        tensor = _auxiliary_stress_local(x, y, mode, K)
        sigma[:, element] = [tensor[0, 0], tensor[1, 1], tensor[0, 1]]
    spacing = 2.0 / (int(round(np.sqrt(len(mesh.nodes)))) - 1)
    damage = np.zeros(len(mesh.nodes))
    damage[
        (mesh.nodes[:, 0] < 0.0)
        & (np.abs(mesh.nodes[:, 1]) < 0.6 * spacing)
    ] = 1.0
    return nodal.ravel(), sigma, damage


@pytest.mark.parametrize(
    "mode,K_expected,cross_name,tolerance",
    [
        ("I", 5.0e6, "K_II_Pa_sqrt_m", 0.03),
        ("II", 3.0e6, "K_I_Pa_sqrt_m", 0.08),
    ],
)
def test_signed_interaction_integral_recovers_williams_field(
    mode, K_expected, cross_name, tolerance
):
    mat = ElasticProperties()
    mesh = _mesh()
    u, sigma, damage = _exact_state(mesh, mode, K_expected, mat)
    result = compute_signed_interaction_integral(
        mesh,
        u,
        sigma,
        damage,
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        mat,
        1.0,
        cfg=SimpleNamespace(r_inner_factor=0.15, r_outer_factor=0.8),
        D=isotropic_plane_strain_D(mat.E, mat.nu),
    )
    recovered = (
        result.K_I_Pa_sqrt_m if mode == "I" else result.K_II_Pa_sqrt_m
    )
    assert recovered == pytest.approx(K_expected, rel=tolerance)
    assert abs(getattr(result, cross_name)) < 0.01 * K_expected
    assert result.diagnostics["signed_modes"] == ["I", "II"]


def test_interaction_integral_fails_closed_for_anisotropic_stiffness():
    mat = ElasticProperties()
    D = isotropic_plane_strain_D(mat.E, mat.nu)
    D[2, 2] *= 1.2
    with pytest.raises(ValueError, match="is isotropic"):
        require_isotropic_stiffness(D, mat)
