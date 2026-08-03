"""Neutral-mode-stabilized affine-DMD propagation for v10.2.30.

Least-squares roundoff can place an exactly neutral cycle-map eigenvalue at
``1 + epsilon``.  Raising that map to 1e12 cycles then creates a false secular
amplification.  This module binds the validated v1 propagator to a fit routine
that snaps only eigenvalues within a configurable neighborhood of unity to
exactly one and verifies that the stabilized map still satisfies the original
training tolerance.
"""
from __future__ import annotations

import math
import os
import types

import numpy as np

from . import persistent_site_high_cycle_dmd_v10230 as _base


MODEL_ID = "v10.2.30_validated_affine_dmd_cycle_map_v2_neutral_stable"


def _neutral_tolerance() -> float:
    try:
        value = float(os.environ.get("V10230_DMD_NEUTRAL_EIGEN_TOL", 1.0e-6))
    except (TypeError, ValueError):
        value = 1.0e-6
    if not math.isfinite(value):
        value = 1.0e-6
    return max(value, 0.0)


def _stabilize_neutral_modes(A: np.ndarray, tolerance: float) -> np.ndarray:
    matrix = np.asarray(A, dtype=float)
    if matrix.size == 0:
        return matrix.copy()
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    adjusted = eigenvalues.copy()
    mask = np.abs(eigenvalues - 1.0) <= float(tolerance)
    adjusted[mask] = 1.0
    if not np.any(mask):
        return matrix.copy()
    try:
        stabilized = eigenvectors @ np.diag(adjusted) @ np.linalg.inv(eigenvectors)
    except np.linalg.LinAlgError:
        return matrix.copy()
    if np.max(np.abs(np.imag(stabilized))) > 1.0e-10:
        return matrix.copy()
    return np.real(stabilized)


def _fit_affine_model_stabilized(states, outputs, config):
    model = _base._fit_affine_model(states, outputs, config)
    A_original = np.asarray(model["A"], dtype=float)
    A = _stabilize_neutral_modes(A_original, _neutral_tolerance())
    coordinates = np.asarray(model["coordinates"], dtype=float)
    previous = coordinates[:, :-1]
    following = coordinates[:, 1:]
    fitted = A @ previous + np.asarray(model["c"], dtype=float)[:, None]
    training_error = _base._relative_vector_error(fitted, following)
    model["A_unstabilized"] = A_original
    model["A"] = A
    model["training_error"] = training_error
    eigenvalues = np.linalg.eigvals(A)
    model["spectral_radius"] = (
        float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    )
    return model


def _bind_propagator():
    base = _base.propagate_dmd_cycles
    namespace = dict(base.__globals__)
    namespace["MODEL_ID"] = MODEL_ID
    namespace["_fit_affine_model"] = _fit_affine_model_stabilized
    function = types.FunctionType(
        base.__code__,
        namespace,
        name=base.__name__,
        argdefs=base.__defaults__,
        closure=base.__closure__,
    )
    function.__kwdefaults__ = base.__kwdefaults__
    function.__annotations__ = dict(base.__annotations__)
    function.__doc__ = base.__doc__
    function.__module__ = __name__
    return function


dmd_config = _base.dmd_config
propagate_dmd_cycles = _bind_propagator()


__all__ = [
    "MODEL_ID",
    "dmd_config",
    "propagate_dmd_cycles",
    "_stabilize_neutral_modes",
]
