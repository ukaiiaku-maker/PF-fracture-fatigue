"""Constitutive-tangent guard for the v9.14 v7 VHCF projective accelerator.

The first 1e14-cycle v7 VHCF run exposed a failure mode that endpoint
coarse/fine agreement alone cannot detect.  A monotonic or increasing state
field can be extrapolated with an almost constant secant rate over an enormous
block; both coarse and fine projectors then agree, while a real exact cycle at
the projected endpoint would produce essentially no further increment.

This module adds no constitutive physics.  It wraps the existing step-doubled
block evaluator and probes one exact authoritative v7 cycle from the accepted
fine midpoint and fine endpoint.  The average rate used to traverse each half
block must agree with the local exact one-cycle constitutive rate at that half's
right endpoint.  The existing ``block_state_rtol`` remains the only tolerance:
the guard folds its maximum active-field rate mismatch into the existing block
state error.

The guarded engine is intended to keep O(log N) VHCF traversal while preventing
unbounded linear extrapolation of source-linked blunting, mobile content, or
other active fields after their per-cycle rates have relaxed.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import math
from typing import Any, Callable

import numpy as np

import v914_v7_vhcf_event_engine as _base_engine
from v914_v7_adaptive_block_accelerator import evaluate_step_doubled_block as _base_evaluate
from v914_v7_cycle_map import advance_v7_cycle
from v914_v7_projective_state import ACTIVE_MONOTONIC_ARRAYS, ACTIVE_NONNEGATIVE_ARRAYS


GUARD_ID = "v9.14_v7_vhcf_active_constitutive_tangent_guard_v1"
ENGINE_ID = "v9.14_v7_vhcf_event_to_event_adaptive_block_tangent_guard_v2"
ACTIVE_FIELDS = ACTIVE_NONNEGATIVE_ARRAYS + ACTIVE_MONOTONIC_ARRAYS

_TANGENT_PROBE_MAPS = 0


def _rate_norm_error(
    left_state,
    right_state,
    local_next_state,
    span_cycles: int,
    name: str,
) -> float:
    """Compare half-block average field rate with exact local endpoint rate."""
    span = int(span_cycles)
    if span < 1:
        raise ValueError("span_cycles must be positive")
    a = np.asarray(getattr(left_state, name), dtype=float)
    b = np.asarray(getattr(right_state, name), dtype=float)
    c = np.asarray(getattr(local_next_state, name), dtype=float)
    if a.shape != b.shape or b.shape != c.shape:
        return math.inf
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)) or np.any(~np.isfinite(c)):
        return math.inf
    average = (b - a) / float(span)
    local = c - b
    numerator = float(np.linalg.norm(average - local))
    denominator = max(
        float(np.linalg.norm(average)),
        float(np.linalg.norm(local)),
        1.0,
    )
    return numerator / denominator


def _active_rate_consistency(
    current_state,
    mid_state,
    end_state,
    mid_probe_state,
    end_probe_state,
    half_span_cycles: int,
) -> dict[str, Any]:
    first: dict[str, float] = {}
    second: dict[str, float] = {}
    for name in ACTIVE_FIELDS:
        first[name] = _rate_norm_error(
            current_state,
            mid_state,
            mid_probe_state,
            half_span_cycles,
            name,
        )
        second[name] = _rate_norm_error(
            mid_state,
            end_state,
            end_probe_state,
            half_span_cycles,
            name,
        )
    maximum = max([0.0, *first.values(), *second.values()])
    return {
        "guard_id": GUARD_ID,
        "half_span_cycles": int(half_span_cycles),
        "first_half_active_rate_error": first,
        "second_half_active_rate_error": second,
        "maximum_active_rate_relative_error": float(maximum),
    }


def evaluate_step_doubled_block_tangent_guarded(
    previous_anchor_state,
    current_anchor_state,
    *,
    previous_anchor_cycle: int,
    current_anchor_cycle: int,
    current_anchor_hazard: float,
    block_stride: int,
    loading,
    cycle_controls,
    cycle_map_fn: Callable[..., tuple[Any, float, dict[str, Any]]] = advance_v7_cycle,
) -> dict[str, Any]:
    """Existing step-doubled block plus two private exact tangent probes."""
    global _TANGENT_PROBE_MAPS
    trial = _base_evaluate(
        previous_anchor_state,
        current_anchor_state,
        previous_anchor_cycle=previous_anchor_cycle,
        current_anchor_cycle=current_anchor_cycle,
        current_anchor_hazard=current_anchor_hazard,
        block_stride=block_stride,
        loading=loading,
        cycle_controls=cycle_controls,
        cycle_map_fn=cycle_map_fn,
    )
    half = int(block_stride) // 2
    mid_probe_state, mid_probe_hazard, _ = cycle_map_fn(
        copy.deepcopy(trial["fine_mid_state"]), loading, cycle_controls
    )
    end_probe_state, end_probe_hazard, _ = cycle_map_fn(
        copy.deepcopy(trial["fine_end_state"]), loading, cycle_controls
    )
    _TANGENT_PROBE_MAPS += 2

    consistency = _active_rate_consistency(
        current_anchor_state,
        trial["fine_mid_state"],
        trial["fine_end_state"],
        mid_probe_state,
        end_probe_state,
        half,
    )
    tangent_error = float(consistency["maximum_active_rate_relative_error"])
    endpoint = dict(trial["endpoint_state_error"])
    endpoint["coarse_fine_maximum_relative_error"] = float(
        endpoint["maximum_relative_error"]
    )
    endpoint["constitutive_tangent_maximum_relative_error"] = tangent_error
    endpoint["maximum_relative_error"] = max(
        float(endpoint["maximum_relative_error"]), tangent_error
    )
    trial["endpoint_state_error"] = endpoint
    trial["constitutive_tangent_guard"] = consistency
    trial["tangent_probe_mid_hazard"] = float(mid_probe_hazard)
    trial["tangent_probe_end_hazard"] = float(end_probe_hazard)
    trial["additional_cycle_map_evaluations"] = 2
    return trial


def tangent_probe_maps() -> int:
    return int(_TANGENT_PROBE_MAPS)


def reset_tangent_probe_maps() -> None:
    global _TANGENT_PROBE_MAPS
    _TANGENT_PROBE_MAPS = 0


def run_v7_vhcf_event_to_event_tangent_guarded(*args, run_controls, **kwargs):
    """Run the unchanged event engine with tangent-guarded block admission.

    The inherited engine accounts three cycle maps per step-doubled block.  The
    guard adds two private probes.  To preserve the user's total map budget, the
    inherited internal budget is conservatively reduced by 3/5 and the returned
    diagnostics are corrected to the actual number of authoritative cycle-map
    evaluations.
    """
    reset_tangent_probe_maps()
    requested_budget = int(run_controls.maximum_cycle_map_evaluations)
    internal_budget = max(1, int(math.floor(3.0 * requested_budget / 5.0)))
    internal_controls = replace(
        run_controls, maximum_cycle_map_evaluations=internal_budget
    )

    old_evaluator = _base_engine.evaluate_step_doubled_block
    old_engine_id = _base_engine.ENGINE_ID
    try:
        _base_engine.evaluate_step_doubled_block = evaluate_step_doubled_block_tangent_guarded
        _base_engine.ENGINE_ID = ENGINE_ID
        result = _base_engine.run_v7_vhcf_event_to_event(
            *args, run_controls=internal_controls, **kwargs
        )
    finally:
        _base_engine.evaluate_step_doubled_block = old_evaluator
        _base_engine.ENGINE_ID = old_engine_id

    probes = tangent_probe_maps()
    inherited = int(result.get("total_cycle_map_evaluations", 0))
    actual = inherited + probes
    result["engine_id"] = ENGINE_ID
    result["tangent_guard_id"] = GUARD_ID
    result["tangent_probe_cycle_map_evaluations"] = probes
    result["inherited_cycle_map_evaluations"] = inherited
    result["total_cycle_map_evaluations"] = actual
    result["requested_total_cycle_map_budget"] = requested_budget
    result["conservative_internal_cycle_map_budget"] = internal_budget
    result["physical_cycles_per_total_cycle_map_evaluation"] = (
        float(result.get("completed_physical_cycles", 0)) / max(float(actual), 1.0)
    )
    identity = dict(result.get("identity_contract", {}))
    identity["engine_id"] = ENGINE_ID
    identity["tangent_guard_id"] = GUARD_ID
    identity["requested_total_cycle_map_budget"] = requested_budget
    result["identity_contract"] = identity
    return result


__all__ = [
    "ACTIVE_FIELDS",
    "ENGINE_ID",
    "GUARD_ID",
    "evaluate_step_doubled_block_tangent_guarded",
    "reset_tangent_probe_maps",
    "run_v7_vhcf_event_to_event_tangent_guarded",
    "tangent_probe_maps",
]
