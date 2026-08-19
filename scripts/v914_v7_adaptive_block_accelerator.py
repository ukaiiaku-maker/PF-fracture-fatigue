"""Adaptive multi-cycle block growth for the v9.14 intrinsic reverse-glide v7 model.

The authoritative within-cycle law remains ``advance_v7_cycle``.  This module
accelerates only between resolved cycle endpoints.

After the existing retrospective readiness gate has passed, a candidate block
of S physical cycles is evaluated by cycle-level step doubling:

coarse:
    project S-1 endpoints, resolve the right anchor with one exact cycle map

fine:
    project S/2-1 endpoints, resolve midpoint anchor
    project S/2-1 endpoints from the midpoint secant, resolve right anchor

The accepted state is always the fine path.  The coarse path is only a local
error estimator.  Block hazard action is integrated from log-linear bridges
between resolved anchor hazards; coarse/fine block actions provide an
independent hazard error estimate.  No material or kinetic parameter is altered.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np

from v914_v7_adaptive_projective_accelerator import readiness_prediction
from v914_v7_cycle_map import advance_v7_cycle, cycle_endpoint_summary
from v914_v7_projective_accelerator import _telemetry_record
from v914_v7_projective_state import (
    ACTIVE_MONOTONIC_ARRAYS,
    ACTIVE_NONNEGATIVE_ARRAYS,
    PROJECTOR_ID,
    project_v7_state_secant,
)


ACCELERATOR_ID = "v9.14_v7_adaptive_block_step_doubling_accelerator_v1"


@dataclass(frozen=True)
class AdaptiveBlockControls:
    minimum_exact_cycles: int = 4
    readiness_relative_tolerance: float = 0.05
    readiness_consecutive_passes: int = 2
    initial_block_stride: int = 4
    maximum_block_stride: int = 64
    block_state_rtol: float = 0.03
    block_hazard_rtol: float = 0.03
    max_projection_constraint_correction: float = 0.10

    def validate(self) -> None:
        if int(self.minimum_exact_cycles) < 3:
            raise ValueError("minimum_exact_cycles must be at least 3")
        if int(self.readiness_consecutive_passes) < 1:
            raise ValueError("readiness_consecutive_passes must be positive")
        for name in (
            "initial_block_stride",
            "maximum_block_stride",
        ):
            value = int(getattr(self, name))
            if value < 4 or value & (value - 1):
                raise ValueError(f"{name} must be a power of two >= 4")
        if int(self.maximum_block_stride) < int(self.initial_block_stride):
            raise ValueError("maximum_block_stride must be >= initial_block_stride")
        for name in (
            "readiness_relative_tolerance",
            "block_state_rtol",
            "block_hazard_rtol",
            "max_projection_constraint_correction",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")


def aggregate_log_bridge_hazard(left: float, right: float, span_cycles: int) -> float:
    """Hazard action for the next ``span_cycles`` cycles from two anchor hazards.

    ``left`` is the resolved hazard for cycle n and ``right`` is the resolved
    hazard at cycle n+span.  Hazards at intervening integer cycles are assumed
    linear in log space.  The returned action includes cycles n+1...n+span,
    including the right anchor and excluding the left anchor.
    """
    span = int(span_cycles)
    if span < 1:
        raise ValueError("span_cycles must be positive")
    floor = 1.0e-300
    a = max(float(left), floor)
    b = max(float(right), floor)
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError("anchor hazards must be finite")
    logq = (math.log(b) - math.log(a)) / float(span)
    if abs(logq) < 1.0e-12:
        return float(span) * a
    # sum_{j=1}^span a exp(j logq)
    return float(a * math.exp(logq) * math.expm1(span * logq) / math.expm1(logq))


def _relative(a: float, b: float, floor: float) -> float:
    return abs(float(a) - float(b)) / max(abs(float(a)), abs(float(b)), float(floor))


def _norm_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    a = np.asarray(reference, dtype=float)
    b = np.asarray(candidate, dtype=float)
    if a.shape != b.shape:
        return math.inf
    return float(np.linalg.norm(b - a)) / max(
        float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1.0
    )


def endpoint_state_error(reference_state, candidate_state, cycle_index: int) -> dict[str, Any]:
    """Compare two candidate cycle-end states without using hazard."""
    field_errors = {
        name: _norm_error(getattr(reference_state, name), getattr(candidate_state, name))
        for name in ACTIVE_NONNEGATIVE_ARRAYS + ACTIVE_MONOTONIC_ARRAYS
    }
    ref = cycle_endpoint_summary(reference_state, 0.0, int(cycle_index))
    cand = cycle_endpoint_summary(candidate_state, 0.0, int(cycle_index))
    floors = {
        "shielding_MPa_sqrt_m": 1.0e-3,
        "mobile_line_content": 1.0,
        "retained_line_content": 1.0,
        "returned_source_slip": 1.0e-3,
        "tip_radius_m": 1.0e-9,
        "cumulative_source_slip": 1.0,
    }
    summary_errors = {
        name: _relative(ref[name], cand[name], floor)
        for name, floor in floors.items()
    }
    maximum = max([0.0, *field_errors.values(), *summary_errors.values()])
    return {
        "maximum_relative_error": float(maximum),
        "full_state_relative_norm_error": field_errors,
        "summary_relative_error": summary_errors,
    }


def _project_then_resolve(
    previous_anchor_state,
    current_anchor_state,
    *,
    anchor_gap_cycles: int,
    span_cycles: int,
    loading,
    cycle_controls,
    cycle_map_fn: Callable[..., tuple[Any, float, dict[str, Any]]],
) -> tuple[Any, float, dict[str, Any], dict[str, Any]]:
    """Advance a block span using projection followed by one exact right anchor."""
    span = int(span_cycles)
    if span < 1:
        raise ValueError("span_cycles must be positive")
    if span == 1:
        projected = copy.deepcopy(current_anchor_state)
        projection = {
            "projector_id": PROJECTOR_ID,
            "maximum_relative_constraint_correction": 0.0,
            "skip_cycles": 0,
        }
    else:
        projected, projection = project_v7_state_secant(
            previous_anchor_state,
            current_anchor_state,
            anchor_gap_cycles=int(anchor_gap_cycles),
            skip_cycles=span - 1,
            frequency_Hz=float(loading.frequency_Hz),
        )
    final_state, final_hazard, telemetry = cycle_map_fn(
        projected, loading, cycle_controls
    )
    return final_state, float(final_hazard), telemetry, projection


def evaluate_step_doubled_block(
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
    """Evaluate one coarse/fine projective block; do not mutate accepted state."""
    stride = int(block_stride)
    if stride < 4 or stride & (stride - 1):
        raise ValueError("block_stride must be a power of two >= 4")
    gap = int(current_anchor_cycle) - int(previous_anchor_cycle)
    if gap < 1:
        raise ValueError("anchor cycle ordering is invalid")
    half = stride // 2

    coarse_state, coarse_hazard, coarse_tel, coarse_proj = _project_then_resolve(
        previous_anchor_state,
        current_anchor_state,
        anchor_gap_cycles=gap,
        span_cycles=stride,
        loading=loading,
        cycle_controls=cycle_controls,
        cycle_map_fn=cycle_map_fn,
    )

    mid_cycle = int(current_anchor_cycle) + half
    mid_state, mid_hazard, mid_tel, mid_proj = _project_then_resolve(
        previous_anchor_state,
        current_anchor_state,
        anchor_gap_cycles=gap,
        span_cycles=half,
        loading=loading,
        cycle_controls=cycle_controls,
        cycle_map_fn=cycle_map_fn,
    )

    fine_state, fine_hazard, fine_tel, fine_proj = _project_then_resolve(
        current_anchor_state,
        mid_state,
        anchor_gap_cycles=half,
        span_cycles=half,
        loading=loading,
        cycle_controls=cycle_controls,
        cycle_map_fn=cycle_map_fn,
    )

    end_cycle = int(current_anchor_cycle) + stride
    state_error = endpoint_state_error(fine_state, coarse_state, end_cycle)
    coarse_action = aggregate_log_bridge_hazard(
        current_anchor_hazard, coarse_hazard, stride
    )
    fine_action = (
        aggregate_log_bridge_hazard(current_anchor_hazard, mid_hazard, half)
        + aggregate_log_bridge_hazard(mid_hazard, fine_hazard, half)
    )
    hazard_error = _relative(coarse_action, fine_action, 1.0e-300)
    correction = max(
        float(coarse_proj.get("maximum_relative_constraint_correction", 0.0)),
        float(mid_proj.get("maximum_relative_constraint_correction", 0.0)),
        float(fine_proj.get("maximum_relative_constraint_correction", 0.0)),
    )

    return {
        "block_stride": stride,
        "start_cycle": int(current_anchor_cycle),
        "mid_cycle": mid_cycle,
        "end_cycle": end_cycle,
        "coarse_state": coarse_state,
        "coarse_end_hazard": coarse_hazard,
        "coarse_telemetry": coarse_tel,
        "fine_mid_state": mid_state,
        "fine_mid_hazard": mid_hazard,
        "fine_mid_telemetry": mid_tel,
        "fine_end_state": fine_state,
        "fine_end_hazard": fine_hazard,
        "fine_end_telemetry": fine_tel,
        "coarse_block_hazard_action": float(coarse_action),
        "fine_block_hazard_action": float(fine_action),
        "block_hazard_relative_error": float(hazard_error),
        "endpoint_state_error": state_error,
        "maximum_projection_constraint_correction": float(correction),
    }


def _largest_power_of_two_at_most(value: int) -> int:
    n = int(value)
    if n < 1:
        return 0
    return 1 << (n.bit_length() - 1)


def run_adaptive_block_multicycle(
    initial_state,
    loading,
    cycle_controls,
    cycles: int,
    *,
    block_controls: AdaptiveBlockControls | None = None,
    cycle_map_fn: Callable[..., tuple[Any, float, dict[str, Any]]] = advance_v7_cycle,
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Resolve the transient exactly, then use adaptive cycle-level step doubling."""
    ncycles = int(cycles)
    if ncycles < 1:
        raise ValueError("cycles must be positive")
    controls = block_controls or AdaptiveBlockControls()
    controls.validate()

    state = copy.deepcopy(initial_state)
    exact_history: list[dict[str, Any]] = []
    block_history: list[dict[str, Any]] = []
    readiness_history: list[dict[str, Any]] = []
    accepted_anchor_telemetry: list[dict[str, Any]] = []
    resolved_states: list[tuple[int, Any, float]] = []

    cycle = 0
    cumulative_hazard = 0.0
    readiness_streak = 0
    promoted = False
    promotion_cycle: int | None = None
    stride = int(controls.initial_block_stride)

    accepted_cycle_map_count = 0
    total_cycle_map_evaluations = 0
    private_error_estimator_cycle_maps = 0
    rejected_block_count = 0
    fallback_exact_count = 0
    accepted_block_count = 0
    maximum_accepted_stride = 0
    maximum_accepted_state_error = 0.0
    maximum_accepted_hazard_error = 0.0
    maximum_projection_correction = 0.0

    while cycle < ncycles:
        if not promoted:
            cycle += 1
            state, hazard, telemetry = cycle_map_fn(state, loading, cycle_controls)
            total_cycle_map_evaluations += 1
            accepted_cycle_map_count += 1
            cumulative_hazard += max(float(hazard), 0.0)
            summary = cycle_endpoint_summary(state, hazard, cycle)
            summary["resolution"] = "shared_cycle_map_readiness"
            summary["cumulative_hazard_action"] = cumulative_hazard
            exact_history.append(summary)
            accepted_anchor_telemetry.append(
                _telemetry_record(cycle, telemetry, summary["resolution"])
            )
            resolved_states.append((cycle, copy.deepcopy(state), float(hazard)))
            if len(resolved_states) > 3:
                resolved_states = resolved_states[-3:]

            if len(resolved_states) == 3:
                c0, s0, _ = resolved_states[0]
                c1, s1, _ = resolved_states[1]
                c2, s2, _ = resolved_states[2]
                if c1 == c0 + 1 and c2 == c1 + 1:
                    readiness = readiness_prediction(
                        s0,
                        s1,
                        s2,
                        frequency_Hz=float(loading.frequency_Hz),
                        cycle_index=c2,
                    )
                    passed = bool(
                        readiness["maximum_relative_error"]
                        <= float(controls.readiness_relative_tolerance)
                        and readiness["projection_constraint_correction"]
                        <= float(controls.max_projection_constraint_correction)
                    )
                    readiness["pass"] = passed
                    readiness_history.append(readiness)
                    readiness_streak = readiness_streak + 1 if passed else 0

            if (
                cycle >= int(controls.minimum_exact_cycles)
                and readiness_streak >= int(controls.readiness_consecutive_passes)
                and ncycles - cycle >= int(controls.initial_block_stride)
            ):
                promoted = True
                if promotion_cycle is None:
                    promotion_cycle = cycle
                stride = int(controls.initial_block_stride)
            continue

        remaining = ncycles - cycle
        if remaining < int(controls.initial_block_stride):
            # Short tail: exact cycles are cheaper and better defined than a block
            # smaller than the qualified step-doubling stencil.
            cycle += 1
            state, hazard, telemetry = cycle_map_fn(state, loading, cycle_controls)
            total_cycle_map_evaluations += 1
            accepted_cycle_map_count += 1
            cumulative_hazard += max(float(hazard), 0.0)
            summary = cycle_endpoint_summary(state, hazard, cycle)
            summary["resolution"] = "shared_cycle_map_exact_tail"
            summary["cumulative_hazard_action"] = cumulative_hazard
            exact_history.append(summary)
            accepted_anchor_telemetry.append(
                _telemetry_record(cycle, telemetry, summary["resolution"])
            )
            resolved_states.append((cycle, copy.deepcopy(state), float(hazard)))
            if len(resolved_states) > 3:
                resolved_states = resolved_states[-3:]
            continue

        if len(resolved_states) < 2:
            raise RuntimeError("adaptive block accelerator lacks two resolved anchors")
        previous_cycle, previous_state, _ = resolved_states[-2]
        current_cycle, current_state, current_hazard = resolved_states[-1]
        if current_cycle != cycle:
            raise RuntimeError("accepted state and current resolved anchor are inconsistent")

        candidate_stride = min(
            int(stride),
            int(controls.maximum_block_stride),
            _largest_power_of_two_at_most(remaining),
        )
        if candidate_stride < int(controls.initial_block_stride):
            promoted = False
            readiness_streak = 0
            continue

        trial = evaluate_step_doubled_block(
            previous_state,
            current_state,
            previous_anchor_cycle=previous_cycle,
            current_anchor_cycle=current_cycle,
            current_anchor_hazard=current_hazard,
            block_stride=candidate_stride,
            loading=loading,
            cycle_controls=cycle_controls,
            cycle_map_fn=cycle_map_fn,
        )
        total_cycle_map_evaluations += 3
        private_error_estimator_cycle_maps += 1
        correction = float(trial["maximum_projection_constraint_correction"])
        state_err = float(trial["endpoint_state_error"]["maximum_relative_error"])
        hazard_err = float(trial["block_hazard_relative_error"])
        maximum_projection_correction = max(maximum_projection_correction, correction)

        passed = bool(
            correction <= float(controls.max_projection_constraint_correction)
            and state_err <= float(controls.block_state_rtol)
            and hazard_err <= float(controls.block_hazard_rtol)
        )

        trial_record = {
            "start_cycle": int(trial["start_cycle"]),
            "mid_cycle": int(trial["mid_cycle"]),
            "end_cycle": int(trial["end_cycle"]),
            "block_stride": int(candidate_stride),
            "accepted": passed,
            "endpoint_state_max_relative_error": state_err,
            "block_hazard_relative_error": hazard_err,
            "coarse_block_hazard_action": float(trial["coarse_block_hazard_action"]),
            "fine_block_hazard_action": float(trial["fine_block_hazard_action"]),
            "maximum_projection_constraint_correction": correction,
            "full_state_relative_norm_error": trial["endpoint_state_error"][
                "full_state_relative_norm_error"
            ],
            "summary_relative_error": trial["endpoint_state_error"][
                "summary_relative_error"
            ],
        }
        block_history.append(trial_record)

        if not passed:
            rejected_block_count += 1
            if candidate_stride > int(controls.initial_block_stride):
                stride = max(
                    int(controls.initial_block_stride),
                    candidate_stride // 2,
                )
                continue

            # Even the minimum projective block was not locally predictable.
            # Discard every private trial state, advance one exact physical cycle,
            # and require readiness again.
            cycle += 1
            state, hazard, telemetry = cycle_map_fn(
                current_state, loading, cycle_controls
            )
            total_cycle_map_evaluations += 1
            accepted_cycle_map_count += 1
            fallback_exact_count += 1
            cumulative_hazard += max(float(hazard), 0.0)
            summary = cycle_endpoint_summary(state, hazard, cycle)
            summary["resolution"] = "shared_cycle_map_block_fallback"
            summary["cumulative_hazard_action"] = cumulative_hazard
            exact_history.append(summary)
            accepted_anchor_telemetry.append(
                _telemetry_record(cycle, telemetry, summary["resolution"])
            )
            resolved_states = [
                (current_cycle, copy.deepcopy(current_state), float(current_hazard)),
                (cycle, copy.deepcopy(state), float(hazard)),
            ]
            promoted = False
            readiness_streak = 0
            stride = int(controls.initial_block_stride)
            continue

        # Fine solution is the only accepted projective trajectory.
        accepted_block_count += 1
        maximum_accepted_stride = max(maximum_accepted_stride, candidate_stride)
        maximum_accepted_state_error = max(maximum_accepted_state_error, state_err)
        maximum_accepted_hazard_error = max(maximum_accepted_hazard_error, hazard_err)
        accepted_cycle_map_count += 2
        cumulative_hazard += max(float(trial["fine_block_hazard_action"]), 0.0)

        mid_cycle = int(trial["mid_cycle"])
        end_cycle = int(trial["end_cycle"])
        mid_state = trial["fine_mid_state"]
        end_state = trial["fine_end_state"]
        mid_hazard = float(trial["fine_mid_hazard"])
        end_hazard = float(trial["fine_end_hazard"])

        mid_summary = cycle_endpoint_summary(mid_state, mid_hazard, mid_cycle)
        end_summary = cycle_endpoint_summary(end_state, end_hazard, end_cycle)
        mid_summary["resolution"] = "adaptive_block_mid_anchor"
        end_summary["resolution"] = "adaptive_block_end_anchor"
        mid_summary["block_stride"] = candidate_stride
        end_summary["block_stride"] = candidate_stride
        end_summary["block_hazard_action"] = float(trial["fine_block_hazard_action"])
        end_summary["cumulative_hazard_action"] = cumulative_hazard
        exact_history.extend((mid_summary, end_summary))
        accepted_anchor_telemetry.extend(
            (
                _telemetry_record(
                    mid_cycle,
                    trial["fine_mid_telemetry"],
                    "adaptive_block_mid_anchor",
                ),
                _telemetry_record(
                    end_cycle,
                    trial["fine_end_telemetry"],
                    "adaptive_block_end_anchor",
                ),
            )
        )

        cycle = end_cycle
        state = copy.deepcopy(end_state)
        resolved_states = [
            (mid_cycle, copy.deepcopy(mid_state), mid_hazard),
            (end_cycle, copy.deepcopy(end_state), end_hazard),
        ]

        # Aggressively test the next power of two.  If it is too large, the
        # step-doubling error estimate rejects it and halves the stride.
        stride = min(candidate_stride * 2, int(controls.maximum_block_stride))

    metadata = {
        "accelerator_id": ACCELERATOR_ID,
        "projector_id": PROJECTOR_ID,
        "cycles": ncycles,
        "promotion_cycle": promotion_cycle,
        "readiness_history": readiness_history,
        "accepted_block_count": accepted_block_count,
        "rejected_block_count": rejected_block_count,
        "fallback_exact_count": fallback_exact_count,
        "maximum_accepted_stride": maximum_accepted_stride,
        "accepted_cycle_map_count": accepted_cycle_map_count,
        "total_cycle_map_evaluations": total_cycle_map_evaluations,
        "private_error_estimator_cycle_maps": private_error_estimator_cycle_maps,
        "accepted_path_cycle_map_speedup": (
            ncycles / max(accepted_cycle_map_count, 1)
        ),
        "actual_cycle_map_evaluation_speedup": (
            ncycles / max(total_cycle_map_evaluations, 1)
        ),
        "maximum_accepted_state_error": maximum_accepted_state_error,
        "maximum_accepted_hazard_error": maximum_accepted_hazard_error,
        "maximum_projection_constraint_correction": maximum_projection_correction,
        "cumulative_hazard_action": cumulative_hazard,
        "block_state_rtol": float(controls.block_state_rtol),
        "block_hazard_rtol": float(controls.block_hazard_rtol),
        "within_cycle_law": "advance_v7_cycle at every accepted resolved anchor",
        "coarse_path_is_private_error_estimator_only": True,
        "accepted_solution": "two_half_blocks_fine_path",
    }
    return state, exact_history, block_history, metadata


__all__ = [
    "ACCELERATOR_ID",
    "AdaptiveBlockControls",
    "aggregate_log_bridge_hazard",
    "endpoint_state_error",
    "evaluate_step_doubled_block",
    "run_adaptive_block_multicycle",
]
