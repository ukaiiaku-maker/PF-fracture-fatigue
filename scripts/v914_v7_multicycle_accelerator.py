"""Exact v7 multi-cycle map and inter-cycle accelerator parity scaffold.

The central contract is

    (state_{n+1}, hazard_n) = F(state_n; loading),

where ``F`` is ``advance_v7_cycle``.  Both the exact multi-cycle path and every
future accelerator anchor call this same map.  There is deliberately no frozen
within-cycle hazard law here.

At this stage cycle skipping is fail-closed.  The module records affine
one-cycle endpoint predictions from previously resolved exact cycles so we can
measure whether an inter-cycle state/hazard predictor is justified before it is
allowed to skip a cycle.
"""
from __future__ import annotations

import copy
from typing import Any, Callable

from v914_v7_cycle_map import advance_v7_cycle, cycle_endpoint_summary


ACCELERATOR_ID = "v9.14_v7_intercycle_accelerator_scaffold_v1"


PREDICTABLE_FIELDS = (
    "hazard_action",
    "shielding_MPa_sqrt_m",
    "mobile_line_content",
    "retained_line_content",
    "returned_source_slip",
    "physical_return_fraction",
    "tip_radius_m",
    "cumulative_source_slip",
    "raw_return_fraction",
)


def affine_next_summary(previous: dict[str, float], current: dict[str, float]) -> dict[str, float]:
    """Predict next endpoint by one-cycle linear extrapolation.

    Diagnostic only.  This never mutates or advances the constitutive state.
    """
    prediction = {"cycle_index": int(current["cycle_index"]) + 1}
    for name in PREDICTABLE_FIELDS:
        prediction[name] = float(current[name]) + (
            float(current[name]) - float(previous[name])
        )
    return prediction


def prediction_errors(
    predicted: dict[str, float], actual: dict[str, float]
) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in PREDICTABLE_FIELDS:
        p = float(predicted[name])
        a = float(actual[name])
        scale = max(abs(p), abs(a), 1.0e-30)
        result[name] = abs(p - a) / scale
    return result


def run_exact_multicycle(
    initial_state,
    loading,
    controls,
    cycles: int,
    *,
    cycle_map_fn: Callable[..., tuple[Any, float, dict[str, Any]]] = advance_v7_cycle,
) -> tuple[Any, list[dict[str, float]], list[dict[str, Any]]]:
    """Resolve consecutive physical cycles using the shared v7 cycle map."""
    if int(cycles) < 1:
        raise ValueError("cycles must be positive")
    state = copy.deepcopy(initial_state)
    records: list[dict[str, float]] = []
    telemetry_records: list[dict[str, Any]] = []
    pending_prediction: dict[str, float] | None = None

    for cycle in range(1, int(cycles) + 1):
        state, hazard, telemetry = cycle_map_fn(state, loading, controls)
        summary = cycle_endpoint_summary(state, hazard, cycle)
        if pending_prediction is not None:
            errors = prediction_errors(pending_prediction, summary)
            for name, value in errors.items():
                summary[f"affine_prediction_relerr_{name}"] = float(value)
        records.append(summary)
        telemetry_records.append(
            {
                "cycle_index": cycle,
                "cycle_map_id": telemetry.get("cycle_map_id"),
                "accepted_intervals": int(telemetry["accepted_intervals"]),
                "refined_intervals": int(telemetry["refined_intervals"]),
                "maximum_depth_reached": int(telemetry["maximum_depth_reached"]),
                "minimum_accepted_phase_width": float(
                    telemetry["minimum_accepted_phase_width"]
                ),
            }
        )
        if len(records) >= 2:
            pending_prediction = affine_next_summary(records[-2], records[-1])

    return state, records, telemetry_records


def run_accelerator_anchor_path(
    initial_state,
    loading,
    controls,
    cycles: int,
    *,
    anchor_stride: int = 1,
    cycle_map_fn: Callable[..., tuple[Any, float, dict[str, Any]]] = advance_v7_cycle,
):
    """Accelerator scaffold that is exact at every resolved anchor.

    ``anchor_stride=1`` is the parity mode and must be identical to the exact
    multi-cycle path.  Larger strides are intentionally rejected until a state
    projection/skip rule has passed prospective error gates against exact cycle
    maps.  This prevents reintroducing the historical weak-T frozen-state error.
    """
    if int(anchor_stride) != 1:
        raise NotImplementedError(
            "cycle skipping is not qualified; accelerator anchors must currently "
            "resolve every cycle through advance_v7_cycle"
        )
    return run_exact_multicycle(
        initial_state,
        loading,
        controls,
        cycles,
        cycle_map_fn=cycle_map_fn,
    )


__all__ = [
    "ACCELERATOR_ID",
    "PREDICTABLE_FIELDS",
    "affine_next_summary",
    "prediction_errors",
    "run_accelerator_anchor_path",
    "run_exact_multicycle",
]
