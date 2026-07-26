from __future__ import annotations

import numpy as np

from arrhenius_fracture import prescribed_geometry_kernel_v10228 as geometry
from arrhenius_fracture.prescribed_geometry_numpy2_compat_v10228 import (
    install_numpy2_orientation_compat,
    orientation_2d,
)


def test_orientation_2d_uses_scalar_determinant():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    assert orientation_2d(a, b, np.array([0.0, 1.0])) == 1.0
    assert orientation_2d(a, b, np.array([0.0, -1.0])) == -1.0
    assert orientation_2d(a, b, np.array([2.0, 0.0])) == 0.0


def test_segment_intersection_runs_with_two_component_vectors():
    install_numpy2_orientation_compat()
    assert geometry._segments_intersect(
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 0.0]),
        1.0e-14,
    )
    assert not geometry._segments_intersect(
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([1.0, 1.0]),
        1.0e-14,
    )
