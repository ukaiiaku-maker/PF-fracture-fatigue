"""Neutral- and output-stabilized affine-DMD propagation for v10.2.30.

Least-squares roundoff can place an exactly neutral cycle-map eigenvalue at
``1 + epsilon`` or give a constant ledger rate a tiny spurious state dependence.
Either error becomes secular when propagated over 1e12 cycles.  This module
snaps only eigenvalues statistically indistinguishable from unity to exactly one
and makes output rows that are constant over the exact training burst exactly
constant.  The stabilized state map is then checked against the unchanged
training and independent endpoint-validation tolerances.
"""
from __future__ import annotations

import math
import os
import types

import numpy as np

from . import persistent_site_high_cycle_dmd_v10230 as _base


MODEL_ID = "v10.2.30_validated_affine_dmd_cycle_map_v2_neutral_output_stable"


def _neutral_tolerance() -> float:
    try:
        value = float(os.environ.get("V10230_DMD_NEUTRAL_EIGEN_TOL", 1.0e-6))
    except (TypeError, ValueError):
        value = 1.0e-6
    if not math.isfinite(value):
        value = 1.0e-6
    return max(value, 0.0)


def _constant_output_tolerance() -> float:
    try:
        value = float(
            os.environ.get("V10230_DMD_CONSTANT_OUTPUT_REL_TOL", 1.0e-12)
        )
    except (TypeError, ValueError):
        value = 1.0e-12
    if not math.isfinite(value):
        value = 1.0e-12
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


def _stabilize_constant_outputs(
    C: np.ndarray,
    d: np.ndarray,
    outputs: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    slopes = np.asarray(C, dtype=float).copy()
    offsets = np.asarray(d, dtype=float).copy()
    observed = np.asarray(outputs, dtype=float)
    constant_rows: list[int] = []
    for index in range(observed.shape[0]):
        row = observed[index]
        scale = max(float(np.max(np.abs(row))), 1.0e-300)
        span = float(np.max(row) - np.min(row))
        if span / scale <= float(tolerance):
            slopes[index, :] = 0.0
            offsets[index] = float(np.mean(row))
            constant_rows.append(index)
    return slopes, offsets, tuple(constant_rows)


def _fit_affine_model_stabilized(states, outputs, config):
    model = _base._fit_affine_model(states, outputs, config)
    A_original = np.asarray(model["A"], dtype=float)
    A = _stabilize_neutral_modes(A_original, _neutral_tolerance())
    coordinates = np.asarray(model["coordinates"], dtype=float)
    previous = coordinates[:, :-1]
    following = coordinates[:, 1:]
    fitted = A @ previous + np.asarray(model["c"], dtype=float)[:, None]
    training_error = _base._relative_vector_error(fitted, following)

    C, d, constant_rows = _stabilize_constant_outputs(
        np.asarray(model["C"], dtype=float),
        np.asarray(model["d"], dtype=float),
        np.asarray(outputs, dtype=float),
        _constant_output_tolerance(),
    )
    output_fitted = C @ previous + d[:, None]
    output_error_by_row = []
    for index in range(outputs.shape[0]):
        output_error_by_row.append(
            _base._relative_vector_error(output_fitted[index], outputs[index])
        )

    model["A_unstabilized"] = A_original
    model["A"] = A
    model["C_unstabilized"] = np.asarray(model["C"], dtype=float)
    model["d_unstabilized"] = np.asarray(model["d"], dtype=float)
    model["C"] = C
    model["d"] = d
    model["constant_output_rows"] = constant_rows
    model["output_training_error_by_row"] = tuple(output_error_by_row)
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
    "_stabilize_constant_outputs",
    "_stabilize_neutral_modes",
]
