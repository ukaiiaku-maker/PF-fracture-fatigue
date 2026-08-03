"""High-cycle propagation operators for v10.2.30.

Two accelerations are supported and both preserve the original constitutive map:

* stationary propagation integrates the already-drawn first-passage threshold
  exactly over whole periodic cycles;
* projective propagation advances a slowly drifting active state only after a
  burst-based tangent and an independently evaluated projected-state map agree.

Neither operator draws RNG state, changes geometry, or imposes a fatigue-growth law.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import os
from typing import Any

import numpy as np

from .persistent_site_high_cycle_state_v10230 import (
    ActiveStateSnapshot,
    apply_ledger_delta,
    capture_stochastic_state,
    project_physical_state,
    residual_metrics,
    restore_active_state,
    serialize_active_state,
    stochastic_state_equal,
)
from .persistent_site_poincare_v10230 import PoincareResult, one_cycle_map


MODEL_ID = "v10.2.30_stationary_and_projective_high_cycle_propagation_v1"


@dataclass
class StationaryPropagationResult:
    cycles_consumed: float
    event_within_guard: bool
    hazard_action_added: float
    normalized_clock_added: float
    ledger_delta: dict[str, float]


@dataclass
class ProjectivePropagationResult:
    accepted: bool
    cycles_consumed: float
    burst_cycles: int
    projected_cycles: float
    hazard_action_added: float
    normalized_clock_added: float
    ledger_delta: dict[str, float]
    state_end: ActiveStateSnapshot
    drift_relative_error: float
    hazard_relative_error: float
    transition_preserved: bool
    attempts: int
    failure_reason: str | None


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


def projective_config() -> dict[str, float | int]:
    return {
        "burst_cycles": _env_int("V10230_PROJECTIVE_BURST_CYCLES", 6, 3),
        "maximum_project_cycles": _env_float(
            "V10230_PROJECTIVE_MAX_CYCLES", 1.0e9, 1.0
        ),
        "minimum_project_cycles": _env_float(
            "V10230_PROJECTIVE_MIN_CYCLES", 10.0, 0.0
        ),
        "initial_factor": _env_float(
            "V10230_PROJECTIVE_INITIAL_FACTOR", 16.0, 1.0
        ),
        "drift_relative_tolerance": _env_float(
            "V10230_PROJECTIVE_DRIFT_REL_TOL", 1.0e-3, 1.0e-12
        ),
        "hazard_relative_tolerance": _env_float(
            "V10230_PROJECTIVE_HAZARD_REL_TOL", 1.0e-3, 1.0e-12
        ),
        "curvature_relative_tolerance": _env_float(
            "V10230_PROJECTIVE_CURVATURE_REL_TOL", 0.1, 1.0e-12
        ),
        "maximum_attempts": _env_int("V10230_PROJECTIVE_MAX_ATTEMPTS", 8, 1),
        "event_guard_cycles": _env_float(
            "V10230_HIGH_CYCLE_EVENT_GUARD_CYCLES", 2.0, 1.0
        ),
    }


def _sum_ledgers(total: dict[str, float], increment: dict[str, float], factor: float = 1.0) -> None:
    for key, value in increment.items():
        total[key] = total.get(key, 0.0) + float(factor) * float(value)


def _update_hazard(engine, action: float) -> float:
    added = max(float(action), 0.0)
    threshold = max(float(getattr(engine, "hazard_threshold_action", 1.0)), 1.0e-300)
    current = max(float(getattr(engine, "hazard_action_current", 0.0)), 0.0)
    new_action = min(current + added, threshold)
    engine.hazard_action_current = new_action
    engine.B = min(new_action / threshold, 1.0)
    engine.hazard_last_progress_rate_s = 0.0
    return (new_action - current) / threshold


def propagate_stationary_cycles(
    engine,
    waveform,
    cycle: PoincareResult,
    cycles_requested: float,
    *,
    event_guard_cycles: float | None = None,
) -> StationaryPropagationResult:
    requested = max(float(cycles_requested), 0.0)
    guard = max(
        float(
            projective_config()["event_guard_cycles"]
            if event_guard_cycles is None
            else event_guard_cycles
        ),
        1.0,
    )
    dH = max(float(cycle.hazard_action_per_cycle), 0.0)
    threshold = max(float(getattr(engine, "hazard_threshold_action", 1.0)), 1.0e-300)
    current = max(float(getattr(engine, "hazard_action_current", 0.0)), 0.0)
    remaining_action = max(threshold - current, 0.0)

    if dH <= 0.0:
        whole = math.floor(requested)
        event_near = False
    else:
        cycles_to_event = remaining_action / dH
        safe_whole = max(math.floor(cycles_to_event - guard), 0)
        whole = min(math.floor(requested), safe_whole)
        event_near = cycles_to_event <= requested + guard

    whole = float(max(whole, 0))
    ledger = {key: whole * value for key, value in cycle.ledger_delta_per_cycle.items()}
    apply_ledger_delta(engine, cycle.ledger_delta_per_cycle, whole)
    normalized = _update_hazard(engine, dH * whole)
    engine.t = float(getattr(engine, "t", 0.0)) + whole * float(waveform.period_s)
    if hasattr(engine, "K_prev"):
        engine.K_prev = float(waveform.Kmax)
    return StationaryPropagationResult(
        cycles_consumed=whole,
        event_within_guard=bool(event_near),
        hazard_action_added=dH * whole,
        normalized_clock_added=normalized,
        ledger_delta=ledger,
    )


def _snapshot_with_vector(template: ActiveStateSnapshot, vector: np.ndarray) -> ActiveStateSnapshot:
    return replace(template, vector=np.asarray(vector, dtype=float).copy())


def _relative_vector_error(observed: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(observed - expected)) / max(
        float(np.linalg.norm(observed)),
        float(np.linalg.norm(expected)),
        1.0,
    )


def propagate_projective_cycles(
    engine,
    controller,
    waveform,
    temperature_K: float,
    cycles_requested: float,
) -> ProjectivePropagationResult:
    config = projective_config()
    requested = max(float(cycles_requested), 0.0)
    initial_stochastic = capture_stochastic_state(engine)
    initial = serialize_active_state(
        engine, waveform=waveform, temperature_K=temperature_K
    )
    states = [initial]
    cycles: list[PoincareResult] = []
    total_ledgers: dict[str, float] = {}
    total_action = 0.0

    for _ in range(int(config["burst_cycles"])):
        result = one_cycle_map(
            engine,
            controller,
            waveform,
            temperature_K,
            state=states[-1],
        )
        if result.transition_signature_start != result.transition_signature_end:
            return ProjectivePropagationResult(
                False, 0.0, len(cycles), 0.0, 0.0, 0.0, {}, initial,
                math.inf, math.inf, False, 0, "discrete_transition_in_burst"
            )
        cycles.append(result)
        states.append(result.state_end)
        total_action += result.hazard_action_per_cycle
        _sum_ledgers(total_ledgers, result.ledger_delta_per_cycle)

    burst_count = len(cycles)
    if burst_count < 3:
        return ProjectivePropagationResult(
            False, 0.0, burst_count, 0.0, 0.0, 0.0, {}, initial,
            math.inf, math.inf, False, 0, "insufficient_burst"
        )

    drifts = [states[index + 1].vector - states[index].vector for index in range(len(states) - 1)]
    drift = 0.5 * (drifts[-1] + drifts[-2])
    curvature = _relative_vector_error(drifts[-1], drifts[-2])
    if curvature > float(config["curvature_relative_tolerance"]):
        return ProjectivePropagationResult(
            False, 0.0, burst_count, 0.0, 0.0, 0.0, {}, initial,
            curvature, math.inf, True, 0, "slow_manifold_curvature_too_large"
        )

    action_rate = 0.5 * (
        cycles[-1].hazard_action_per_cycle + cycles[-2].hazard_action_per_cycle
    )
    ledger_rate: dict[str, float] = {}
    for key in set(cycles[-1].ledger_delta_per_cycle) | set(cycles[-2].ledger_delta_per_cycle):
        ledger_rate[key] = 0.5 * (
            cycles[-1].ledger_delta_per_cycle.get(key, 0.0)
            + cycles[-2].ledger_delta_per_cycle.get(key, 0.0)
        )

    available = max(requested - burst_count, 0.0)
    proposed = min(
        available,
        float(config["maximum_project_cycles"]),
        max(float(config["minimum_project_cycles"]), float(config["initial_factor"]) * burst_count),
    )
    threshold = max(float(getattr(engine, "hazard_threshold_action", 1.0)), 1.0e-300)
    current_action = max(float(getattr(engine, "hazard_action_current", 0.0)), 0.0)
    remaining_action = max(threshold - current_action - total_action, 0.0)
    if action_rate > 0.0:
        proposed = min(
            proposed,
            max(
                remaining_action / action_rate - float(config["event_guard_cycles"]),
                0.0,
            ),
        )

    attempts = 0
    while proposed >= float(config["minimum_project_cycles"]) and attempts < int(config["maximum_attempts"]):
        attempts += 1
        vector = project_physical_state(
            states[-1], states[-1].vector + proposed * drift
        )
        projected = _snapshot_with_vector(states[-1], vector)
        verification = one_cycle_map(
            engine,
            controller,
            waveform,
            temperature_K,
            state=projected,
        )
        observed_drift = verification.state_end.vector - verification.state_start.vector
        drift_error = _relative_vector_error(observed_drift, drift)
        hazard_error = abs(verification.hazard_action_per_cycle - action_rate) / max(
            abs(verification.hazard_action_per_cycle), abs(action_rate), 1.0e-300
        )
        transition_ok = (
            verification.transition_signature_start
            == cycles[-1].transition_signature_end
            == verification.transition_signature_end
        )
        if (
            drift_error <= float(config["drift_relative_tolerance"])
            and hazard_error <= float(config["hazard_relative_tolerance"])
            and transition_ok
        ):
            restore_active_state(engine, projected)
            apply_ledger_delta(engine, total_ledgers)
            apply_ledger_delta(engine, ledger_rate, proposed)
            normalized = _update_hazard(engine, total_action + proposed * action_rate)
            total_cycles = float(burst_count) + proposed
            engine.t = float(getattr(engine, "t", 0.0)) + total_cycles * float(waveform.period_s)
            if hasattr(engine, "K_prev"):
                engine.K_prev = float(waveform.Kmax)
            if not stochastic_state_equal(
                initial_stochastic,
                {
                    **capture_stochastic_state(engine),
                    "B": initial_stochastic["B"],
                    "hazard_action_current": initial_stochastic["hazard_action_current"],
                },
            ):
                raise RuntimeError("projective propagation changed threshold or RNG state")
            ledger_total = dict(total_ledgers)
            _sum_ledgers(ledger_total, ledger_rate, proposed)
            return ProjectivePropagationResult(
                True,
                total_cycles,
                burst_count,
                proposed,
                total_action + proposed * action_rate,
                normalized,
                ledger_total,
                projected,
                drift_error,
                hazard_error,
                transition_ok,
                attempts,
                None,
            )
        proposed *= 0.5

    return ProjectivePropagationResult(
        False,
        0.0,
        burst_count,
        0.0,
        0.0,
        0.0,
        {},
        initial,
        drift_error if "drift_error" in locals() else math.inf,
        hazard_error if "hazard_error" in locals() else math.inf,
        transition_ok if "transition_ok" in locals() else False,
        attempts,
        "projective_validation_failed",
    )


__all__ = [
    "MODEL_ID",
    "ProjectivePropagationResult",
    "StationaryPropagationResult",
    "projective_config",
    "propagate_projective_cycles",
    "propagate_stationary_cycles",
]
