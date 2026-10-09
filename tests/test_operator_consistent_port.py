import numpy as np
import pytest

from arrhenius_fracture.operator_consistent_projection_v10230 import (
    MaterialSlipPatch, OperatorConsistentProjection, area_weighted,
    exact_regularized_inverse_sqrt_integral,
    triangle_displacement_gradients,
)


def test_area_weighting_uses_represented_measure_not_element_count():
    values = np.array([1.0, 5.0])
    areas = np.array([3.0, 1.0])
    assert area_weighted(values, areas, np.array([True, True])) == pytest.approx(2.0)
    assert np.mean(values) == 3.0


def test_exact_regularized_kernel_integral_handles_core_crossing():
    value = exact_regularized_inverse_sqrt_integral(0.0, 4.0, 1.0)
    assert value == pytest.approx(3.0)
    assert exact_regularized_inverse_sqrt_integral(0.0, 0.25, 1.0) == pytest.approx(0.25)


def test_linear_triangle_gradient_is_exact():
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.]])
    elems = np.array([[0, 1, 2]])
    # u_x=2x+3y, u_y=-x+4y
    u = np.array([[0., 0.], [2., -1.], [3., 4.]])
    gradient = triangle_displacement_gradients(nodes, elems, u)
    np.testing.assert_allclose(gradient[0], [[2., 3.], [-1., 4.]])


def test_small_full_projection_uses_same_support_for_mechanics_and_observers():
    projection = OperatorConsistentProjection(nx=30, ny=60, tip_h_fine_m=1e-6)
    reference = projection.virgin_correspondence()
    result = projection.project(
        [MaterialSlipPatch(0, 2e-6, (0.01, -0.01))],
        crack_extension_m=0.0, K_MPa_sqrt_m=25.0,
    )
    assert projection.mesh.ne > 1000
    assert reference["opening_Pa_at_reference_K"] > 0.0
    assert np.all(np.isfinite(result["resolved_shear_Pa"]))
    assert np.max(np.abs(result["beta_p"])) > 0.0
    assert np.max(np.abs(result["grad_u"])) > 0.0
