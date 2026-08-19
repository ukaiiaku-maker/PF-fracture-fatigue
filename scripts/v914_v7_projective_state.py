"""Constrained full-state projection for the v9.14 intrinsic reverse-glide v7 cycle map.

The accelerator is a numerical approximation to repeated applications of the
same authoritative cycle map.  This module therefore projects only *between*
resolved cycle endpoints.  It does not define an alternative within-cycle law.

Projection is deliberately conservative in scope.  The fixed-crack fatigue
qualification path may project the spatial mobile, retained, source-slip, and
returned-slip fields together with cumulative bookkeeping ledgers.  Geometry,
material parameters, kernels, and source-law parameters are never projected.
Any crack-extension change causes the projection to fail closed.
"""
from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np


PROJECTOR_ID = "v9.14_v7_full_state_secant_projector_v1"

# These fields feed subsequent constitutive cycles directly.
ACTIVE_NONNEGATIVE_ARRAYS = (
    "mobile_m2",
    "retained_m2",
)

# In a fixed-crack interval these fields are cumulative local ledgers.  Crack
# advance translates them, so projection is prohibited across extension change.
ACTIVE_MONOTONIC_ARRAYS = (
    "accumulated_slip_m2",
    "returned_slip_m2",
)

# Cumulative physical/accounting arrays.  They do not define a new constitutive
# law, but carrying them prevents an accelerated state from silently losing
# return/emission bookkeeping.  The reverse-driven return ledger is required so
# the physical-return diagnostic remains consistent with the projected
# returned-slip/blunting state.
MONOTONIC_LEDGER_ARRAYS = (
    "cumulative_source_activations",
    "cumulative_line_content",
    "cumulative_returned_mobile_per_m",
    "cumulative_reverse_driven_returned_mobile_per_m",
    "cumulative_escaped_mobile_per_m",
    "cumulative_cancelled_slip_line_content",
)

# Positive cumulative scalars.
MONOTONIC_LEDGER_SCALARS = (
    "effective_plastic_dissipation_J_per_m",
    "cumulative_transport_channel_time_s",
    "cumulative_reverse_channel_time_s",
    "cumulative_mobile_exposure_m2_s",
    "cumulative_reverse_mobile_exposure_m2_s",
)

# Signed work ledgers can legitimately increase or decrease.
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
    """Return the geometry/layout quantities that must not change while skipping."""
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
    """Project a fixed-crack v7 cycle-end state along its endpoint secant.

    ``previous_state`` and ``current_state`` are resolved cycle endpoints separated
    by ``anchor_gap_cycles`` physical cycles.  The returned state represents the
    endpoint ``skip_cycles`` cycles beyond ``current_state``.

    The method projects the complete spatial constitutive fields needed by the
    next exact cycle map.  Nonnegative fields are clipped at zero.  Source-slip,
    returned-slip, and cumulative ledgers are constrained to remain monotone for
    this fixed-crack interval.  Returned slip is additionally bounded pointwise
    by accumulated source slip.  The physical time increment is imposed exactly
    from the loading frequency rather than inferred from a fitted slope.
    """
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

    for name in ACTIVE_NONNEGATIVE_ARRAYS:
        if not (hasattr(previous_state, name) and hasattr(current_state, name)):
            raise RuntimeError(f"required v7 active field is missing: {name}")
        value, correction = _secant_array(
            previous_state,
            current_state,
            name,
            factor,
            nonnegative=True,
            monotonic=False,
        )
        setattr(projected, name, value)
        corrections[name] = correction

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

    # Surface return cannot cancel more source-linked slip than has been emitted.
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

    # Time is a kinematic loading coordinate.  Advance it exactly for skipped
    # physical cycles rather than treating it as an extrapolated constitutive field.
    projected.time_s = _finite_scalar(current_state, "time_s") + float(skip) / freq

    # Fixed-crack projective acceleration must not alter these invariants.
    projected.extension_m = float(current_state.extension_m)
    if hasattr(projected, "source_available_m2"):
        projected.source_available_m2 = np.asarray(
            current_state.source_available_m2, dtype=float
        ).copy()
    if hasattr(projected, "source_capacity_m2"):
        projected.source_capacity_m2 = np.asarray(
            current_state.source_capacity_m2, dtype=float
        ).copy()

    # Fail closed on a malformed projected mechanical state.
    for name in ACTIVE_NONNEGATIVE_ARRAYS + ACTIVE_MONOTONIC_ARRAYS:
        value = np.asarray(getattr(projected, name), dtype=float)
        if np.any(~np.isfinite(value)) or np.any(value < 0.0):
            raise RuntimeError(f"projected field is nonphysical: {name}")
    if np.any(projected.returned_slip_m2 > projected.accumulated_slip_m2):
        raise RuntimeError("projected returned slip exceeds accumulated source slip")

    diagnostics = {
        "projector_id": PROJECTOR_ID,
        "anchor_gap_cycles": gap,
        "skip_cycles": skip,
        "secant_factor": factor,
        "maximum_relative_constraint_correction": max(corrections.values(), default=0.0),
        "relative_constraint_corrections": corrections,
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
