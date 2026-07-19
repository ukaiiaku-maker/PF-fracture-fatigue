from types import SimpleNamespace

import numpy as np
import pytest

from arrhenius_fracture.config import ElasticProperties
from arrhenius_fracture.interaction_integral_v1029 import (
    _auxiliary_displacement_local,
    _auxiliary_gradient_local_analytic,
    _auxiliary_stress_local,
    _hermite_plateau_q,
    compute_signed_interaction_integral,
    isotropic_plane_strain_D,
)


def _finite_difference_gradient(x, y, mode, mat):
    r = np.hypot(x, y)
    h = 2.0e-7 * r
    xp = _auxiliary_displacement_local(x + h, y, mode, mat.E, mat.nu)
    xm = _auxiliary_displacement_local(x - h, y, mode, mat.E, mat.nu)
    yp = _auxiliary_displacement_local(x, y + h, mode, mat.E, mat.nu)
    ym = _auxiliary_displacement_local(x, y - h, mode, mat.E, mat.nu)
    return np.column_stack([(xp - xm) / (2.0 * h), (yp - ym) / (2.0 * h)])


@pytest.mark.parametrize("mode", ["I", "II"])
@pytest.mark.parametrize("point", [(0.7, 0.2), (0.4, -0.3), (-0.2, 0.35)])
def test_analytic_auxiliary_gradient_matches_independent_difference(mode, point):
    mat = ElasticProperties()
    exact = _auxiliary_gradient_local_analytic(*point, mode, mat.E, mat.nu)
    check = _finite_difference_gradient(*point, mode, mat)
    assert exact == pytest.approx(check, rel=2.0e-7, abs=2.0e-15)


def test_hermite_weight_is_flat_at_both_annulus_boundaries():
    ri, ro = 0.2, 0.8
    h = 1.0e-7
    values = _hermite_plateau_q(
        np.array([ri - h, ri, ri + h, ro - h, ro, ro + h]), ri, ro
    )
    assert values[0] == pytest.approx(1.0)
    assert values[1] == pytest.approx(1.0)
    assert values[4] == pytest.approx(0.0)
    assert values[5] == pytest.approx(0.0)
    assert abs((values[2] - values[1]) / h) < 2.0e-6
    assert abs((values[4] - values[3]) / h) < 2.0e-6


def _mesh(n=41, radius=1.0):
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
            [[1.0, p[0, 0], p[0, 1]], [1.0, p[1, 0], p[1, 1]], [1.0, p[2, 0], p[2, 1]]]
        )
        inverse = np.linalg.inv(interpolation)
        dNdx[index, 0] = inverse[1]
        dNdx[index, 1] = inverse[2]
        area[index] = 0.5 * abs(
            np.linalg.det(np.column_stack([p[1] - p[0], p[2] - p[0]]))
        )
    return SimpleNamespace(nodes=nodes, elems=elems, dNdx_e=dNdx, area_e=area, ne=len(elems))


def _exact_state(mesh, mode, K, mat):
    nodal = np.zeros((len(mesh.nodes), 2))
    for index, (x, y) in enumerate(mesh.nodes):
        if np.hypot(x, y) > 1.0e-14:
            nodal[index] = _auxiliary_displacement_local(x, y, mode, mat.E, mat.nu, K)
    centroids = mesh.nodes[mesh.elems].mean(axis=1)
    sigma = np.zeros((3, mesh.ne))
    for element, (x, y) in enumerate(centroids):
        tensor = _auxiliary_stress_local(x, y, mode, K)
        sigma[:, element] = [tensor[0, 0], tensor[1, 1], tensor[0, 1]]
    spacing = 2.0 / (int(round(np.sqrt(len(mesh.nodes)))) - 1)
    damage = np.zeros(len(mesh.nodes))
    damage[(mesh.nodes[:, 0] < 0.0) & (np.abs(mesh.nodes[:, 1]) < 0.6 * spacing)] = 1.0
    return nodal.ravel(), sigma, damage


@pytest.mark.parametrize(
    "mode,K_expected,recovery_tolerance",
    [("I", 5.0e6, 0.025), ("II", 3.0e6, 0.05)],
)
def test_hardened_integral_recovers_williams_field_and_is_contour_stable(
    mode, K_expected, recovery_tolerance
):
    mat = ElasticProperties()
    mesh = _mesh()
    u, sigma, damage = _exact_state(mesh, mode, K_expected, mat)
    values = []
    for outer in (0.65, 0.75, 0.85):
        result = compute_signed_interaction_integral(
            mesh,
            u,
            sigma,
            damage,
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            mat,
            1.0,
            cfg=SimpleNamespace(r_inner_factor=0.15, r_outer_factor=outer),
            D=isotropic_plane_strain_D(mat.E, mat.nu),
        )
        recovered = result.K_I_Pa_sqrt_m if mode == "I" else result.K_II_Pa_sqrt_m
        cross = result.K_II_Pa_sqrt_m if mode == "I" else result.K_I_Pa_sqrt_m
        # The analytic-gradient unit test above is stringent. This tolerance is
        # for a coarse linear-triangle discretization of a singular Williams field;
        # mode II converges more slowly and is checked separately from the formula.
        assert recovered == pytest.approx(K_expected, rel=recovery_tolerance)
        assert abs(cross) < 0.01 * K_expected
        assert result.diagnostics["auxiliary_displacement_gradient"] == "analytic_polar_chain_rule"
        assert result.diagnostics["domain_weight"] == "cubic_Hermite_C1"
        values.append(recovered)
    assert (max(values) - min(values)) / K_expected < 0.025
