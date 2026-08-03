"""Chained validated affine-DMD high-cycle propagation for v10.2.30.

A single high-cycle state-machine operation may contain multiple independently
validated affine-DMD segments.  Each segment is trained from a fresh exact
Poincare burst at the current committed state, validated by an independent exact
cycle at its proposed endpoint, and committed transactionally by the underlying
v2 DMD operator.  The next proposed horizon grows from the last accepted horizon
instead of repeatedly requesting the complete remaining 1e12-cycle interval.

This removes the v2 scheduler pathology in which every accepted projective move
was followed by a separate exact burst and another excessively large proposal.
No validation tolerance is relaxed and no unvalidated state is committed.
"""
from __future__ import annotations

import math
import os
from typing import Any

from .persistent_site_high_cycle_dmd_v10230_v2 import (
    MODEL_ID as SEGMENT_MODEL_ID,
    dmd_config,
    propagate_dmd_cycles as propagate_dmd_segment,
)
from .persistent_site_high_cycle_propagation_v10230 import (
    ProjectivePropagationResult,
)
from .persistent_site_high_cycle_state_v10230 import (
    serialize_active_state,
)


MODEL_ID = "v10.2.30_chained_validated_affine_dmd_cycle_map_v3"


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = float(default)
    if not math.isfinite(value):
        value = float(default)
    return max(value, minimum)


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = int(default)
    return max(value, minimum)


def chained_dmd_config() -> dict[str, float | int]:
    return {
        "maximum_segments": _env_int("V10230_DMD_CHAIN_MAX_SEGMENTS", 96, 1),
        "growth_factor": _env_float("V10230_DMD_CHAIN_GROWTH_FACTOR", 2.0, 1.0),
        "minimum_growth_factor": _env_float(
            "V10230_DMD_CHAIN_MIN_GROWTH_FACTOR", 1.25, 1.0
        ),
        "maximum_project_cycles": _env_float(
            "V10230_PROJECTIVE_MAX_CYCLES", 1.0e12, 1.0
        ),
        "minimum_remaining_cycles": _env_float(
            "V10230_DMD_CHAIN_MIN_REMAINING_CYCLES", 1.0, 0.0
        ),
    }


def _sum_mapping(target: dict[str, float], source: dict[str, float]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0.0) + float(value)


def _first_rejected_upper(requested: float, accepted: float, attempts: int) -> float:
    """Infer the nearest rejected upper bound from the v2 halving search."""
    request = max(float(requested), 0.0)
    accepted_value = max(float(accepted), 0.0)
    if int(attempts) <= 1 or accepted_value <= 0.0:
        return math.inf
    return min(request, 2.0 * accepted_value)


def _next_proposal(
    accepted: float,
    rejected_upper: float,
    remaining: float,
    config: dict[str, float | int],
) -> float:
    accepted_value = max(float(accepted), 0.0)
    if accepted_value <= 0.0:
        return 0.0
    growth = float(config["growth_factor"])
    minimum_growth = float(config["minimum_growth_factor"])
    if math.isfinite(rejected_upper) and rejected_upper > accepted_value:
        # Use a logarithmic bracket step.  This remains above the accepted lower
        # bound but avoids repeating the immediately rejected upper horizon.
        proposal = math.sqrt(accepted_value * rejected_upper)
        proposal = max(proposal, minimum_growth * accepted_value)
    else:
        proposal = growth * accepted_value
    return min(
        proposal,
        max(float(remaining), 0.0),
        float(config["maximum_project_cycles"]),
    )


def propagate_chained_dmd_cycles(
    engine,
    controller,
    waveform,
    temperature_K: float,
    cycles_requested: float,
    *,
    requested_project_cycles: float | None = None,
) -> ProjectivePropagationResult:
    requested = max(float(cycles_requested), 0.0)
    segment_config = dmd_config()
    chain_config = chained_dmd_config()
    initial_state = serialize_active_state(
        engine, waveform=waveform, temperature_K=temperature_K
    )
    minimum_project = float(segment_config["minimum_project_cycles"])
    if requested <= 0.0:
        return ProjectivePropagationResult(
            True, 0.0, 0, 0.0, 0.0, 0.0, {}, initial_state,
            0.0, 0.0, True, 0, None
        )

    proposal = max(
        minimum_project,
        float(
            requested_project_cycles
            if requested_project_cycles is not None
            else minimum_project
        ),
    )
    proposal = min(
        proposal,
        requested,
        float(chain_config["maximum_project_cycles"]),
    )

    total_cycles = 0.0
    total_burst_cycles = 0
    total_projected_cycles = 0.0
    total_action = 0.0
    total_normalized = 0.0
    total_ledgers: dict[str, float] = {}
    total_attempts = 0
    last_state = initial_state
    maximum_state_error = 0.0
    maximum_hazard_error = 0.0
    transition_preserved = True
    last_failure: str | None = None
    accepted_segments = 0
    last_accepted_project = 0.0
    last_rejected_upper = math.inf
    segment_audit: list[dict[str, Any]] = []

    while (
        total_cycles < requested
        and accepted_segments < int(chain_config["maximum_segments"])
    ):
        remaining = requested - total_cycles
        if remaining < float(chain_config["minimum_remaining_cycles"]):
            break
        local_proposal = min(proposal, remaining)
        result = propagate_dmd_segment(
            engine,
            controller,
            waveform,
            temperature_K,
            remaining,
            requested_project_cycles=local_proposal,
        )
        total_attempts += int(result.attempts)
        segment_audit.append(
            {
                "requested_project_cycles": float(local_proposal),
                "accepted": bool(result.accepted),
                "cycles_consumed": float(result.cycles_consumed),
                "burst_cycles": int(result.burst_cycles),
                "projected_cycles": float(result.projected_cycles),
                "state_error": float(result.drift_relative_error),
                "hazard_error": float(result.hazard_relative_error),
                "attempts": int(result.attempts),
                "failure_reason": result.failure_reason,
            }
        )
        if not result.accepted or result.cycles_consumed <= 0.0:
            last_failure = result.failure_reason or "dmd_segment_rejected"
            break

        accepted_segments += 1
        total_cycles += float(result.cycles_consumed)
        total_burst_cycles += int(result.burst_cycles)
        total_projected_cycles += float(result.projected_cycles)
        total_action += float(result.hazard_action_added)
        total_normalized += float(result.normalized_clock_added)
        _sum_mapping(total_ledgers, dict(result.ledger_delta))
        last_state = result.state_end
        maximum_state_error = max(
            maximum_state_error, float(result.drift_relative_error)
        )
        maximum_hazard_error = max(
            maximum_hazard_error, float(result.hazard_relative_error)
        )
        transition_preserved = bool(
            transition_preserved and result.transition_preserved
        )

        accepted_project = max(float(result.projected_cycles), minimum_project)
        rejected_upper = _first_rejected_upper(
            local_proposal, accepted_project, int(result.attempts)
        )
        last_accepted_project = accepted_project
        last_rejected_upper = rejected_upper
        remaining = requested - total_cycles
        if remaining <= 0.0:
            break
        proposal = _next_proposal(
            accepted_project,
            rejected_upper,
            remaining,
            chain_config,
        )
        if proposal < minimum_project:
            last_failure = "remaining_horizon_below_dmd_minimum"
            break

    completed = total_cycles >= requested - 1.0e-12 * max(requested, 1.0)
    if completed:
        total_cycles = requested
        last_failure = None
    elif accepted_segments >= int(chain_config["maximum_segments"]):
        last_failure = "dmd_chain_segment_budget_exhausted"

    result = ProjectivePropagationResult(
        accepted=bool(total_cycles > 0.0),
        cycles_consumed=float(total_cycles),
        burst_cycles=int(total_burst_cycles),
        projected_cycles=float(total_projected_cycles),
        hazard_action_added=float(total_action),
        normalized_clock_added=float(total_normalized),
        ledger_delta=total_ledgers,
        state_end=last_state,
        drift_relative_error=float(maximum_state_error),
        hazard_relative_error=float(maximum_hazard_error),
        transition_preserved=bool(transition_preserved),
        attempts=int(total_attempts),
        failure_reason=last_failure,
    )
    # Optional audit attributes are intentionally attached without changing the
    # shared result dataclass used by the v2 reference solver.
    result.chained_dmd = True
    result.chained_dmd_model_id = MODEL_ID
    result.segment_model_id = SEGMENT_MODEL_ID
    result.segments = segment_audit
    result.accepted_segments = int(accepted_segments)
    result.requested_cycles = float(requested)
    result.completed_requested_horizon = bool(completed)
    result.last_accepted_project_cycles = float(last_accepted_project)
    result.last_rejected_upper_cycles = float(last_rejected_upper)
    return result


__all__ = [
    "MODEL_ID",
    "SEGMENT_MODEL_ID",
    "chained_dmd_config",
    "propagate_chained_dmd_cycles",
]
