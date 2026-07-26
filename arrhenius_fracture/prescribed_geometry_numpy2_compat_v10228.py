"""NumPy 2 compatibility for the v10.2.28 prescribed-geometry builder.

NumPy 2 no longer accepts two-component vectors in ``numpy.cross``.  Segment
orientation in the prescribed crack builder is intrinsically two-dimensional,
so use the equivalent scalar determinant explicitly.  This changes no mechanics
or geometry policy; it only restores the intended 2-D computational operation.
"""
from __future__ import annotations

import numpy as np

MODEL_ID = "v10.2.28_numpy2_scalar_2d_orientation_compat_v1"


def orientation_2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the signed twice-area determinant for three 2-D points."""
    av = np.asarray(a, dtype=float).reshape(2)
    bv = np.asarray(b, dtype=float).reshape(2)
    cv = np.asarray(c, dtype=float).reshape(2)
    ab = bv - av
    ac = cv - av
    return float(ab[0] * ac[1] - ab[1] * ac[0])


def install_numpy2_orientation_compat() -> None:
    """Install the scalar 2-D determinant in the direct geometry module."""
    from . import prescribed_geometry_kernel_v10228 as target

    target._orientation = orientation_2d


__all__ = [
    "MODEL_ID",
    "orientation_2d",
    "install_numpy2_orientation_compat",
]
