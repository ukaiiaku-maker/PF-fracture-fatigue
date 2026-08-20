"""Validated rate-separated affine-DMD blocks for reversible v7 VHCF.

Every training and validation cycle calls the authoritative ``advance_v7_cycle``.
Only mobile, retained, and net source-blunting fields enter the affine map.
Gross source emission/return and diagnostic totals are integrated separately
from exact local cycle rates.
"""
from __future__ import annotations

import copy
import math
import os
from typing import Any

import numpy as np

from v914_v7_cycle_map import advance_v7_cycle
from v914_v7_rate_separated_state import (
    MONOTONE_LEDGER_FIELDS,
    apply_ledger_delta,
    capture_ledgers,
    geometry_signature,
    independent_cycle,
    restore_active_state,
    serialize_active_state,
)


MODEL_ID = "v9.14_v7_rate_separated_positive_affine_dmd_v1"


def _env_int(name: str, default: int, minimum: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(value, minimum)


def _fit(states: list[np.ndarray], outputs: list[float], maximum_rank: int = 12) -> dict[str, np.ndarray | float]:
    matrix = np.column_stack(states)
    reference = matrix[:, 0]
    centered = matrix - reference[:, None]
    scale = np.maximum(np.max(np.abs(centered), axis=1), np.maximum(np.abs(reference), 1.0))
    normalized = centered / scale[:, None]
    previous = normalized[:, :-1]
    following = normalized[:, 1:]
    u, singular, _vh = np.linalg.svd(previous, full_matrices=False)
    if singular.size == 0:
        raise ValueError("empty DMD training burst")
    cutoff = max(float(singular[0]) * 1.0e-12, 1.0e-15)
    rank = min(max(int(np.sum(singular > cutoff)), 1), int(maximum_rank))
    basis = u[:, :rank]
    x = basis.T @ previous
    y = basis.T @ following
    design = np.vstack([x, np.ones(x.shape[1])])
    coefficient = np.linalg.lstsq(design.T, y.T, rcond=None)[0].T
    A = coefficient[:, :rank]
    c = coefficient[:, rank]
    eigenvalues, eigenvectors = np.linalg.eig(A)
    adjusted = eigenvalues.copy()
    neutral = np.abs(eigenvalues - 1.0) <= 1.0e-6
    adjusted[neutral] = 1.0
    # Growth faster than a neutral affine mode is unsafe over VHCF horizons.
    unstable = np.abs(adjusted) > 1.0 + 1.0e-8
    adjusted[unstable] /= np.abs(adjusted[unstable])
    if np.any(neutral | unstable):
        try:
            candidate = eigenvectors @ np.diag(adjusted) @ np.linalg.inv(eigenvectors)
            if np.max(np.abs(np.imag(candidate))) <= 1.0e-9:
                A = np.real(candidate)
        except np.linalg.LinAlgError:
            pass
    fitted = A @ x + c[:, None]
    error = float(np.linalg.norm(fitted - y)) / max(float(np.linalg.norm(y)), 1.0)
    observed = np.asarray(outputs, dtype=float)
    output_center = float(np.mean(observed))
    variation = observed - output_center
    if float(np.ptp(observed)) <= 1.0e-12 * max(float(np.max(np.abs(observed))), 1.0e-300):
        C = np.zeros(rank)
        d = output_center
    else:
        output_scale = max(float(np.max(np.abs(variation))), 1.0e-300)
        coefficient_out = np.linalg.lstsq(
            np.vstack([x, np.ones(x.shape[1])]).T,
            variation / output_scale,
            rcond=None,
        )[0]
        C = output_scale * coefficient_out[:rank]
        d = output_center + output_scale * float(coefficient_out[rank])
    return {"reference": reference, "scale": scale, "basis": basis, "A": A, "c": c, "C": C, "d": d, "training_error": error}


def _project(model, start: np.ndarray, cycles: int) -> np.ndarray:
    reference = np.asarray(model["reference"])
    scale = np.asarray(model["scale"])
    basis = np.asarray(model["basis"])
    A = np.asarray(model["A"])
    c = np.asarray(model["c"])
    rank = A.shape[0]
    augmented = np.zeros((rank + 1, rank + 1))
    augmented[:rank, :rank] = A
    augmented[:rank, -1] = c
    augmented[-1, -1] = 1.0
    z = basis.T @ ((np.asarray(start) - reference) / scale)
    propagated = np.linalg.matrix_power(augmented, int(cycles)) @ np.r_[z, 1.0]
    raw = reference + scale * (basis @ propagated[:rank])
    # Positivity map from v5: retain valid values and smoothly map only invalid ones.
    invalid = raw < 0.0
    if np.any(invalid):
        baseline = np.maximum(np.asarray(start), 0.0)
        mapped = baseline * np.exp(np.clip((raw - baseline) / np.maximum(baseline, 1.0), -700.0, 0.0))
        raw[invalid] = np.where(baseline[invalid] > 0.0, mapped[invalid], 0.0)
    if np.any(~np.isfinite(raw)):
        raise FloatingPointError("non-finite v7 DMD projection")
    return raw


def _predict_output(model, vector: np.ndarray) -> float:
    z = np.asarray(model["basis"]).T @ (
        (np.asarray(vector) - np.asarray(model["reference"])) / np.asarray(model["scale"])
    )
    return max(float(np.asarray(model["C"]) @ z + float(model["d"])), 0.0)


def _relative_vector(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))) / max(
        float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1.0
    )


def _relative_scalar(a: float, b: float) -> float:
    return abs(float(a) - float(b)) / max(abs(float(a)), abs(float(b)), 1.0e-300)


def _combine_rates(start, midpoint, endpoint, cycles: int):
    result = {}
    span = 0.0
    for key in set(start) | set(midpoint) | set(endpoint):
        values = [np.asarray(row.get(key, 0.0), dtype=float) for row in (start, midpoint, endpoint)]
        rate = (values[0] + 4.0 * values[1] + values[2]) / 6.0
        if key in MONOTONE_LEDGER_FIELDS:
            rate = np.maximum(rate, 0.0)
        result[key] = float(cycles) * rate
        local_span = np.max(np.stack(values), axis=0) - np.min(np.stack(values), axis=0)
        span = max(span, float(np.linalg.norm(local_span)) / max(*(float(np.linalg.norm(v)) for v in values), 1.0))
    return result, span


def evaluate_rate_separated_dmd_block(
    current_state,
    *,
    block_stride: int,
    loading,
    cycle_controls,
    state_rtol: float,
    hazard_rtol: float,
    cycle_map_fn=advance_v7_cycle,
) -> dict[str, Any]:
    """Return a private DMD block in the shape consumed by the VHCF engine."""
    stride = int(block_stride)
    burst = min(_env_int("V914_V7_DMD_BURST_CYCLES", 12, 4), stride - 2)
    if stride <= burst + 1:
        return {"numerical_pass": False, "failure_reason": "insufficient_dmd_horizon", "cycle_map_evaluations": 0}
    geometry = geometry_signature(current_state)
    exact_state = copy.deepcopy(current_state)
    snapshots = [serialize_active_state(exact_state)]
    hazards = []
    exact_ledger_total = {}
    evaluations = 0
    for _ in range(burst):
        end, hazard, _telemetry, snapshot, delta = independent_cycle(
            exact_state, loading, cycle_controls, cycle_map_fn
        )
        evaluations += 1
        exact_state = end
        snapshots.append(snapshot)
        hazards.append(max(hazard, 0.0))
        for key, value in delta.items():
            exact_ledger_total[key] = np.asarray(exact_ledger_total.get(key, 0.0)) + np.asarray(value)
    model = _fit([row.vector for row in snapshots], hazards, maximum_rank=_env_int("V914_V7_DMD_MAX_RANK", 12, 1))
    remaining = stride - burst
    midpoint_remaining = max(remaining // 2, 1)
    start_vector = snapshots[-1].vector
    midpoint_vector = _project(model, start_vector, midpoint_remaining)
    endpoint_vector = _project(model, start_vector, remaining)
    midpoint_state = copy.deepcopy(exact_state)
    endpoint_state = copy.deepcopy(exact_state)
    restore_active_state(midpoint_state, snapshots[-1], midpoint_vector)
    restore_active_state(endpoint_state, snapshots[-1], endpoint_vector)

    start_end, start_hazard, _st, start_next, start_rate = independent_cycle(exact_state, loading, cycle_controls, cycle_map_fn)
    mid_end, mid_hazard, _mt, mid_next, mid_rate = independent_cycle(midpoint_state, loading, cycle_controls, cycle_map_fn)
    end_end, end_hazard, _et, end_next, end_rate = independent_cycle(endpoint_state, loading, cycle_controls, cycle_map_fn)
    evaluations += 3
    start_prediction = _project(model, start_vector, 1)
    mid_prediction = _project(model, midpoint_vector, 1)
    end_prediction = _project(model, endpoint_vector, 1)
    state_error = max(
        _relative_vector(start_next.vector, start_prediction),
        _relative_vector(mid_next.vector, mid_prediction),
        _relative_vector(end_next.vector, end_prediction),
    )
    # Fit hazard as an independently sampled local output, never as frozen state.
    predicted_mid_hazard = _predict_output(model, midpoint_vector)
    predicted_end_hazard = _predict_output(model, endpoint_vector)
    hazard_error = max(
        _relative_scalar(mid_hazard, predicted_mid_hazard),
        _relative_scalar(end_hazard, predicted_end_hazard),
    )
    projected_action = float(remaining) * (start_hazard + 4.0 * mid_hazard + end_hazard) / 6.0
    upper_action = float(remaining) * max(start_hazard, mid_hazard, end_hazard)
    ledger_delta, ledger_span = _combine_rates(start_rate, mid_rate, end_rate, remaining)
    for key, value in exact_ledger_total.items():
        ledger_delta[key] = np.asarray(ledger_delta.get(key, 0.0)) + np.asarray(value)

    final_state = copy.deepcopy(current_state)
    apply_ledger_delta(final_state, ledger_delta)
    restore_active_state(final_state, serialize_active_state(final_state), endpoint_vector)
    final_state.time_s = float(current_state.time_s) + float(stride) / float(loading.frequency_Hz)
    if geometry_signature(final_state) != geometry:
        raise RuntimeError("v7 DMD block changed crack geometry")
    numerical_pass = bool(
        float(model["training_error"]) <= float(state_rtol)
        and state_error <= float(state_rtol)
        and hazard_error <= float(hazard_rtol)
    )
    return {
        "numerical_pass": numerical_pass,
        "failure_reason": None if numerical_pass else "dmd_validation_failed",
        "block_stride": stride,
        "fine_end_state": final_state,
        "fine_mid_state": midpoint_state,
        "fine_mid_hazard": float(mid_hazard),
        "fine_end_hazard": float(end_hazard),
        "fine_block_hazard_action": float(sum(hazards) + projected_action),
        "upper_block_hazard_action": float(sum(hazards) + upper_action),
        "endpoint_state_error": {"maximum_relative_error": state_error},
        "block_hazard_relative_error": hazard_error,
        "maximum_projection_constraint_correction": 0.0,
        "training_error": float(model["training_error"]),
        "maximum_ledger_rate_span": ledger_span,
        "cycle_map_evaluations": evaluations,
        "model_id": MODEL_ID,
    }


__all__ = ["MODEL_ID", "evaluate_rate_separated_dmd_block"]
