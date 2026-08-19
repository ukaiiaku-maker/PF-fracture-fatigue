"""Constrained full-state projection for the v9.14 intrinsic reverse-glide v7 cycle map.

The accelerator is a numerical approximation to repeated applications of the
same authoritative cycle map. This module therefore projects only *between*
resolved cycle endpoints. It does not define an alternative within-cycle law.

Projection is deliberately conservative in scope. The fixed-crack fatigue
qualification path may project the spatial mobile, retained, source-slip, and
returned-slip fields together with cumulative bookkeeping ledgers. Geometry,
material parameters, kernels, and source-law parameters are never projected.
Any crack-extension change causes the projection to fail closed.

The mobile and retained fields require a positivity-preserving predictor. A raw
linear secant can extrapolate a decaying nonnegative density through zero even
when the underlying transport/reaction relaxation remains physical. For active
components that decreased between resolved anchors, v2 therefore extrapolates
the endpoint logarithmic slope. Increasing components retain the ordinary
linear secant. This is parameter-free and changes only the inter-cycle numerical
predictor. Monotonic source-slip and cumulative ledgers retain constrained linear
secants.
"""
from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np


PROJECTOR_ID = "v9.14_v7_full_state_positive_decay_secant_projector_v2"

ACTIVE_NONNEGATIVE_ARRAYS = (
    "mobile_m2",
    "retained_m2",
)

ACTIVE_MONOTONIC_ARRAYS = (
    "accumulated_slip_m2",
    "returned_slip_m2",
)

MONOTONIC_LEDGER_ARRAYS = (
    "cumulative_source_activations",
    "cumulative_line_content",
    "cumulative_returned_mobile_per_m",
    "cumulative_escaped_mobile_per_m",
    "cumulative_cancelled_slip_line_content",
)

MONOTONIC_LEDGER_SCALARS = (
    "effective_plastic_dissipation_J_per_m",
    "cumulative_transport_channel_time_s",
    "cumulative_reverse_channel_time_s",
    "cumulative_mobile_exposure_m2_s",
    "cumulative_reverse_mobile_exposure_m2_s",
)

SIGNED_LEDGER_SCALARS = (
    "external_plastic_work_J_per_m",
    "nonlocal_shielding_work_J_per_m",
    "internal_stress_work_J_per_m",
    "effective_plastic_work_J_per_m",
)


def _finite_array(owner: Any, name: str) -> np.ndarray:
    value = np.asarray(getattr(owner, name), dtype=float)
    if np.any(~np.isfinite(value)):
        raise RuntimeError(f"non-finite state field: {name}")
    return value


def _finite_scalar(owner: Any, name: str) -> float:
    value = float(getattr(owner, name))
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite state scalar: {name}")
    return value


def fixed_geometry_signature(state) -> tuple[Any, ...]:
    return (
        int(state.c.n_systems),
        int(state.c.n_bins),
        tuple(np.shape(state.mobile_m2)),
        tuple(np.shape(state.retained_m2)),
        float(state.dx),
        float(state.c.mpz_length_m),
        float(state.extension_m),
    )


def _relative_correction(raw: np.ndarray, corrected: np.ndarray) -> float:
    numerator = float(np.linalg.norm(np.asarray(corrected) - np.asarray(raw)))
    denominator = max(
        float(np.linalg.norm(np.asarray(raw))),
        float(np.linalg.norm(np.asarray(corrected))),
        1.0,
    )
    return numerator / denominator


def _secant_array(
    previous,
    current,
    name: str,
    factor: float,
    *,
    nonnegative: bool,
    monotonic: bool,
) -> tuple[np.ndarray, float]:
    a = _finite_array(previous, name)
    b = _finite_array(current, name)
    if a.shape != b.shape:
        raise RuntimeError(f"state shape changed for {name}: {a.shape} -> {b.shape}")
    raw = b + float(factor) * (b - a)
    corrected = raw.copy()
    if nonnegative:
        corrected = np.maximum(corrected, 0.0)
    if monotonic:
        corrected = np.maximum(corrected, b)
    return corrected, _relative_correction(raw, corrected)


def _positive_decay_secant_array(
    previous,
    current,
    name: str,
    factor: float,
) -> tuple[np.ndarray, float]:
    """Project a nonnegative active field without linear zero-crossing artifacts.

    Components that are stationary or increasing use the ordinary linear secant.
    Components that decreased between anchors use their endpoint logarithmic
    slope. Exact zeros remain zero. The second return value measures departure
    from the unconstrained linear secant for diagnostics only; it is not a
    physical constraint correction.
    """
    a = _finite_array(previous, name)
    b = _finite_array(current, name)
    if a.shape != b.shape:
        raise RuntimeError(f"state shape changed for {name}: {a.shape} -> {b.shape}")
    if np.any(a < 0.0) or np.any(b < 0.0):
        raise RuntimeError(f"active nonnegative field contains negative values: {name}")

    f = float(factor)
    linear = b + f * (b - a)
    projected = linear.copy()

    decreasing_positive = (b < a) & (b > 0.0) & (a > 0.0)
    if np.any(decreasing_positive):
        log_ratio = np.log(b[decreasing_positive] / a[decreasing_positive])
        projected[decreasing_positive] = (
            b[decreasing_positive] * np.exp(f * log_ratio)
        )

    decreasing_to_zero = (b < a) & (b <= 0.0)
    projected[decreasing_to_zero] = 0.0
    projected = np.maximum(projected, 0.0)
    if np.any(~np.isfinite(projected)):
        raise RuntimeError(f"positive-decay projection produced non-finite values: {name}")
    return projected, _relative_correction(linear, projected)


def _secant_scalar(
    previous,
    current,
    name: str,
    factor: float,
    *,
    monotonic: bool,
) -> tuple[float, float]:
    a = _finite_scalar(previous, name)
    b = _finite_scalar(current, name)
    raw = b + float(factor) * (b - a)
    corrected = max(raw, b, 0.0) if monotonic else raw
    scale = max(abs(raw), abs(corrected), 1.0)
    return float(corrected), abs(float(corrected) - float(raw)) / scale


def project_v7_state_secant(
    previous_state,
    current_state,
    *,
    anchor_gap_cycles: int,
    skip_cycles: int,
    frequency_Hz: float,
) -> tuple[Any, dict[str, Any]]:
    """Project a fixed-crack v7 cycle-end state along its endpoint trend."""
    gap = int(anchor_gap_cycles)
    skip = int(skip_cycles)
    freq = float(frequency_Hz)
    if gap < 1:
        raise ValueError("anchor_gap_cycles must be positive")
    if skip < 1:
        raise ValueError("skip_cycles must be positive")
    if not math.isfinite(freq) or freq <= 0.0:
        raise ValueError("frequency_Hz must be positive and finite")

    sig_prev = fixed_geometry_signature(previous_state)
    sig_cur = fixed_geometry_signature(current_state)
    if sig_prev[:-1] != sig_cur[:-1]:
        raise RuntimeError("cannot project across a state-grid/geometry-layout change")
    extension_scale = max(abs(sig_prev[-1]), abs(sig_cur[-1]), 1.0e-30)
    if abs(sig_cur[-1] - sig_prev[-1]) > 1.0e-12 * extension_scale:
        raise RuntimeError("cannot project across crack extension; resolve the event exactly")

    factor = float(skip) / float(gap)
    projected = copy.deepcopy(current_state)
    corrections: dict[str, float] = {}
    predictor_departures: dict[str, float] = {}

    for name in ACTIVE_NONNEGATIVE_ARRAYS:
        if not (hasattr(previous_state, name) and hasattr(current_state, name)):
            raise RuntimeError(f"required v7 active field is missing: {name}")
        value, departure = _positive_decay_secant_array(
            previous_state,
            current_state,
            name,
            factor,
        )
        setattr(projected, name, value)
        corrections[name] = 0.0
        predictor_departures[name] = departure

    for name in ACTIVE_MONOTONIC_ARRAYS:
        if not (hasattr(previous_state, name) and hasattr(current_state, name)):
            raise RuntimeError(f"required v7 active field is missing: {name}")
        value, correction = _secant_array(
            previous_state,
            current_state,
            name,
            factor,
            nonnegative=True,
            monotonic=True,
        )
        setattr(projected, name, value)
        corrections[name] = correction

    returned_before_cap = np.asarray(projected.returned_slip_m2, dtype=float).copy()
    projected.returned_slip_m2 = np.minimum(
        np.maximum(projected.returned_slip_m2, 0.0),
        np.maximum(projected.accumulated_slip_m2, 0.0),
    )
    corrections["returned_slip_pointwise_cap"] = _relative_correction(
        returned_before_cap, projected.returned_slip_m2
    )

    for name in MONOTONIC_LEDGER_ARRAYS:
        if not (hasattr(previous_state, name) and hasattr(current_state, name)):
            continue
        value, correction = _secant_array(
            previous_state,
            current_state,
            name,
            factor,
            nonnegative=True,
            monotonic=True,
        )
        setattr(projected, name, value)
        corrections[name] = correction

    for name in MONOTONIC_LEDGER_SCALARS:
        if not (hasattr(previous_state, name) and hasattr(current_state, name)):
            continue
        value, correction = _secant_scalar(
            previous_state, current_state, name, factor, monotonic=True
        )
        setattr(projected, name, value)
        corrections[name] = correction

    for name in SIGNED_LEDGER_SCALARS:
        if not (hasattr(previous_state, name) and hasattr(current_state, name)):
            continue
        value, correction = _secant_scalar(
            previous_state, current_state, name, factor, monotonic=False
        )
        setattr(projected, name, value)
        corrections[name] = correction

    projected.time_s = _finite_scalar(current_state, "time_s") + float(skip) / freq
    projected.extension_m = float(current_state.extension_m)
    if hasattr(projected, "source_available_m2"):
        projected.source_available_m2 = np.asarray(
            current_state.source_available_m2, dtype=float
        ).copy()
    if hasattr(projected, "source_capacity_m2"):
        projected.source_capacity_m2 = np.asarray(
            current_state.source_capacity_m2, dtype=float
        ).copy()

    for name in ACTIVE_NONNEGATIVE_ARRAYS + ACTIVE_MONOTONIC_ARRAYS:
        value = np.asarray(getattr(projected, name), dtype=float)
        if np.any(~np.isfinite(value)) or np.any(value < 0.0):
            raise RuntimeError(f"projected field is nonphysical: {name}")
    if np.any(projected.returned_slip_m2 > projected.accumulated_slip_m2):
        raise RuntimeError("projected returned slip exceeds accumulated source slip")

    diagnostics = {
        "projector_id": PROJECTOR_ID,
        "active_nonnegative_predictor": "linear_growth_logarithmic_decay_secant",
        "anchor_gap_cycles": gap,
        "skip_cycles": skip,
        "secant_factor": factor,
        "maximum_relative_constraint_correction": max(corrections.values(), default=0.0),
        "relative_constraint_corrections": corrections,
        "maximum_active_predictor_departure_from_linear": max(
            predictor_departures.values(), default=0.0
        ),
        "active_predictor_departure_from_linear": predictor_departures,
        "time_s": float(projected.time_s),
        "extension_m": float(projected.extension_m),
    }
    return projected, diagnostics


__all__ = [
    "PROJECTOR_ID",
    "ACTIVE_NONNEGATIVE_ARRAYS",
    "ACTIVE_MONOTONIC_ARRAYS",
    "MONOTONIC_LEDGER_ARRAYS",
    "MONOTONIC_LEDGER_SCALARS",
    "SIGNED_LEDGER_SCALARS",
    "fixed_geometry_signature",
    "project_v7_state_secant",
]
