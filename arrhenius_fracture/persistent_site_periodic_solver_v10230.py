"""Accelerated periodic-state solver for the v10.2.30 Poincare map.

The solver applies Anderson-accelerated fixed-point iteration to the complete
active-state vector.  Every map evaluation is performed on a private engine
clone, so solver iterations consume neither physical cycles, cumulative ledgers,
first-passage action, nor RNG state.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import os
from typing import Any

import numpy as np

from .persistent_site_high_cycle_state_v10230 import (
    ActiveStateSnapshot,
    ResidualMetrics,
    project_physical_state,
    residual_metrics,
    serialize_active_state,
    state_distance,
)
from .persistent_site_poincare_v10230 import PoincareResult, one_cycle_map


MODEL_ID = "v10.2.30_anderson_periodic_state_solver_v1"


@dataclass
class PeriodicSolverResult:
    converged: bool
    state: ActiveStateSnapshot
    cycle: PoincareResult
    residual: ResidualMetrics
    iterations: int
    map_evaluations: int
    distance_from_initial: float
    history: tuple[dict[str, Any], ...]
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


def periodic_config() -> dict[str, float | int]:
    return {
        "relative_tolerance": _env_float(
            "V10230_PERIODIC_RELATIVE_TOL", 1.0e-8, 1.0e-14
        ),
        "diagnostic_tolerance": _env_float(
            "V10230_PERIODIC_DIAGNOSTIC_TOL", 1.0e-6, 1.0e-14
        ),
        "maximum_iterations": _env_int(
            "V10230_PERIODIC_MAX_ITERATIONS", 40, 2
        ),
        "anderson_depth": _env_int("V10230_PERIODIC_ANDERSON_DEPTH", 6, 1),
        "damping": min(
            _env_float("V10230_PERIODIC_DAMPING", 0.8, 1.0e-6), 1.0
        ),
        "regularization": _env_float(
            "V10230_PERIODIC_REGULARIZATION", 1.0e-12, 0.0
        ),
        "divergence_factor": _env_float(
            "V10230_PERIODIC_DIVERGENCE_FACTOR", 100.0, 1.0
        ),
    }


def _snapshot_with_vector(template: ActiveStateSnapshot, vector: np.ndarray) -> ActiveStateSnapshot:
    return replace(template, vector=np.asarray(vector, dtype=float).copy())


def _anderson_candidate(
    x_history: list[np.ndarray],
    f_history: list[np.ndarray],
    y: np.ndarray,
    depth: int,
    regularization: float,
) -> np.ndarray:
    count = min(int(depth), len(f_history) - 1)
    if count < 1:
        return y
    start = len(f_history) - count - 1
    dF = np.column_stack(
        [f_history[index + 1] - f_history[index] for index in range(start, len(f_history) - 1)]
    )
    dX = np.column_stack(
        [x_history[index + 1] - x_history[index] for index in range(start, len(x_history) - 1)]
    )
    current_f = f_history[-1]
    if dF.size == 0:
        return y
    lhs = dF.T @ dF
    if regularization > 0.0:
        lhs = lhs + regularization * np.eye(lhs.shape[0])
    rhs = dF.T @ current_f
    try:
        coefficients = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(dF, current_f, rcond=None)[0]
    return y - (dX + dF) @ coefficients


def solve_periodic_state(
    engine,
    controller,
    waveform,
    temperature_K: float,
    *,
    initial_state: ActiveStateSnapshot | None = None,
) -> PeriodicSolverResult:
    config = periodic_config()
    initial = initial_state or serialize_active_state(
        engine, waveform=waveform, temperature_K=temperature_K
    )
    x = initial.vector.copy()
    x_history: list[np.ndarray] = []
    f_history: list[np.ndarray] = []
    audit: list[dict[str, Any]] = []
    best_residual = math.inf
    best_state = initial
    best_cycle = one_cycle_map(
        engine, controller, waveform, temperature_K, state=initial
    )
    best_metrics = residual_metrics(
        best_cycle.state_start,
        best_cycle.state_end,
        relative_tolerance=float(config["relative_tolerance"]),
        diagnostic_tolerance=float(config["diagnostic_tolerance"]),
    )
    evaluations = 1

    for iteration in range(1, int(config["maximum_iterations"]) + 1):
        candidate_state = _snapshot_with_vector(
            initial, project_physical_state(initial, x)
        )
        cycle = one_cycle_map(
            engine, controller, waveform, temperature_K, state=candidate_state
        )
        evaluations += 1
        metrics = residual_metrics(
            cycle.state_start,
            cycle.state_end,
            relative_tolerance=float(config["relative_tolerance"]),
            diagnostic_tolerance=float(config["diagnostic_tolerance"]),
        )
        if cycle.transition_signature_start != cycle.transition_signature_end:
            return PeriodicSolverResult(
                False,
                cycle.state_start,
                cycle,
                metrics,
                iteration,
                evaluations,
                state_distance(initial, cycle.state_start),
                tuple(audit),
                "discrete_transition_inside_cycle",
            )

        audit.append(
            {
                "iteration": iteration,
                "maximum_relative_residual": metrics.maximum_relative,
                "rms_relative_residual": metrics.rms_relative,
                "distance_from_initial": state_distance(initial, cycle.state_start),
                "hazard_action_per_cycle": cycle.hazard_action_per_cycle,
            }
        )
        if metrics.maximum_relative < best_residual:
            best_residual = metrics.maximum_relative
            best_state = cycle.state_start
            best_cycle = cycle
            best_metrics = metrics
        if metrics.converged:
            fixed = cycle.state_end
            verified = one_cycle_map(
                engine, controller, waveform, temperature_K, state=fixed
            )
            evaluations += 1
            verification = residual_metrics(
                verified.state_start,
                verified.state_end,
                relative_tolerance=float(config["relative_tolerance"]),
                diagnostic_tolerance=float(config["diagnostic_tolerance"]),
            )
            if verification.converged:
                return PeriodicSolverResult(
                    True,
                    verified.state_start,
                    verified,
                    verification,
                    iteration,
                    evaluations,
                    state_distance(initial, verified.state_start),
                    tuple(audit),
                    None,
                )

        f = cycle.state_end.vector - cycle.state_start.vector
        x_history.append(cycle.state_start.vector.copy())
        f_history.append(f.copy())
        y = cycle.state_end.vector.copy()
        accelerated = _anderson_candidate(
            x_history,
            f_history,
            y,
            int(config["anderson_depth"]),
            float(config["regularization"]),
        )
        damping = float(config["damping"])
        next_x = (1.0 - damping) * cycle.state_start.vector + damping * accelerated
        x = project_physical_state(initial, next_x)

        if (
            iteration > 2
            and metrics.maximum_relative
            > float(config["divergence_factor"]) * max(best_residual, 1.0e-30)
        ):
            return PeriodicSolverResult(
                False,
                best_state,
                best_cycle,
                best_metrics,
                iteration,
                evaluations,
                state_distance(initial, best_state),
                tuple(audit),
                "periodic_iteration_diverged",
            )

    return PeriodicSolverResult(
        False,
        best_state,
        best_cycle,
        best_metrics,
        int(config["maximum_iterations"]),
        evaluations,
        state_distance(initial, best_state),
        tuple(audit),
        "maximum_iterations",
    )


__all__ = [
    "MODEL_ID",
    "PeriodicSolverResult",
    "periodic_config",
    "solve_periodic_state",
]
