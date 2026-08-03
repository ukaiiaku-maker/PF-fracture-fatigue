"""Validated reduced affine cycle-map propagation for v10.2.30.

The ordinary one-cycle Poincare map is sampled on a private trajectory.  A
low-rank affine map is fitted in the observed state subspace, including a
constant coordinate so neutral linear-drift modes are represented exactly.
Hazard action and cumulative ledger increments are propagated through the same
augmented map.  A proposed endpoint is committed only after an independent
exact-cycle validation of state drift, hazard rate, ledger rates, and discrete
transition signatures.

No stochastic threshold is drawn or changed, no crack geometry is altered, and
no empirical fatigue-growth law is introduced.
"""
from __future__ import annotations

from dataclasses import replace
import math
import os
from typing import Any

import numpy as np

from .persistent_site_high_cycle_propagation_v10230 import (
    ProjectivePropagationResult,
)
from .persistent_site_high_cycle_state_v10230 import (
    ActiveStateSnapshot,
    apply_ledger_delta,
    capture_stochastic_state,
    geometry_signature,
    project_physical_state,
    restore_active_state,
)
from .persistent_site_poincare_v10230 import one_cycle_map


MODEL_ID = "v10.2.30_validated_affine_dmd_cycle_map_v1"


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


def dmd_config() -> dict[str, float | int]:
    return {
        "burst_cycles": _env_int("V10230_DMD_BURST_CYCLES", 24, 6),
        "maximum_rank": _env_int("V10230_DMD_MAX_RANK", 12, 1),
        "singular_relative_cutoff": _env_float(
            "V10230_DMD_SINGULAR_REL_CUTOFF", 1.0e-10, 0.0
        ),
        "training_relative_tolerance": _env_float(
            "V10230_DMD_TRAINING_REL_TOL", 5.0e-4, 1.0e-12
        ),
        "state_validation_relative_tolerance": _env_float(
            "V10230_DMD_STATE_VALIDATION_REL_TOL", 1.0e-3, 1.0e-12
        ),
        "hazard_validation_relative_tolerance": _env_float(
            "V10230_DMD_HAZARD_VALIDATION_REL_TOL", 1.0e-3, 1.0e-12
        ),
        "ledger_validation_relative_tolerance": _env_float(
            "V10230_DMD_LEDGER_VALIDATION_REL_TOL", 2.0e-3, 1.0e-12
        ),
        "physical_projection_relative_tolerance": _env_float(
            "V10230_DMD_PHYSICAL_PROJECTION_REL_TOL", 1.0e-6, 0.0
        ),
        "minimum_project_cycles": _env_int(
            "V10230_DMD_MIN_PROJECT_CYCLES", 32, 1
        ),
        "maximum_attempts": _env_int("V10230_DMD_MAX_ATTEMPTS", 20, 1),
        "event_guard_cycles": _env_float(
            "V10230_HIGH_CYCLE_EVENT_GUARD_CYCLES", 2.0, 1.0
        ),
    }


def _relative_vector_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))) / max(
        float(np.linalg.norm(a)), float(np.linalg.norm(b)), 1.0
    )


def _relative_scalar_error(a: float, b: float, floor: float = 1.0e-300) -> float:
    return abs(float(a) - float(b)) / max(abs(float(a)), abs(float(b)), floor)


def _snapshot_with_vector(
    template: ActiveStateSnapshot,
    vector: np.ndarray,
    *,
    diagnostics: dict[str, float] | None = None,
) -> ActiveStateSnapshot:
    return replace(
        template,
        vector=np.asarray(vector, dtype=float).copy(),
        diagnostics=(
            dict(template.diagnostics) if diagnostics is None else dict(diagnostics)
        ),
    )


def _update_hazard(engine, action: float) -> float:
    added = max(float(action), 0.0)
    threshold = max(float(getattr(engine, "hazard_threshold_action", 1.0)), 1.0e-300)
    current = max(float(getattr(engine, "hazard_action_current", 0.0)), 0.0)
    new_action = min(current + added, threshold)
    engine.hazard_action_current = new_action
    engine.B = min(new_action / threshold, 1.0)
    engine.hazard_last_progress_rate_s = 0.0
    return (new_action - current) / threshold


def _stochastic_preserved_except_clock(before: dict[str, Any], after: dict[str, Any]) -> bool:
    adjusted = dict(after)
    adjusted["B"] = before["B"]
    adjusted["hazard_action_current"] = before["hazard_action_current"]
    return adjusted == before


def _fit_affine_model(
    states: list[ActiveStateSnapshot],
    outputs: np.ndarray,
    config: dict[str, float | int],
):
    vectors = np.column_stack([state.vector for state in states])
    reference = vectors[:, 0].copy()
    deviations = vectors - reference[:, None]
    sample = deviations[:, 1:]
    if sample.size == 0:
        raise ValueError("DMD fit requires a nonempty trajectory")
    U, singular, _vh = np.linalg.svd(sample, full_matrices=False)
    if singular.size == 0 or singular[0] <= 0.0:
        rank = 1
        U = np.zeros((vectors.shape[0], 1), dtype=float)
    else:
        retained = singular >= singular[0] * float(config["singular_relative_cutoff"])
        rank = min(
            int(config["maximum_rank"]),
            int(np.count_nonzero(retained)),
            states.__len__() - 1,
        )
        rank = max(rank, 1)
        U = U[:, :rank]

    coordinates = U.T @ deviations
    previous = coordinates[:, :-1]
    following = coordinates[:, 1:]
    design = np.vstack([previous, np.ones(previous.shape[1], dtype=float)])
    coefficient = np.linalg.lstsq(design.T, following.T, rcond=None)[0].T
    A = coefficient[:, :rank]
    c = coefficient[:, rank]

    output_coefficient = np.linalg.lstsq(design.T, outputs.T, rcond=None)[0].T
    C = output_coefficient[:, :rank]
    d = output_coefficient[:, rank]

    fitted = A @ previous + c[:, None]
    training_error = _relative_vector_error(fitted, following)
    eigenvalues = np.linalg.eigvals(A)
    spectral_radius = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    return {
        "reference": reference,
        "basis": U,
        "coordinates": coordinates,
        "A": A,
        "c": c,
        "C": C,
        "d": d,
        "rank": rank,
        "training_error": training_error,
        "spectral_radius": spectral_radius,
    }


def _propagate_augmented(model: dict[str, Any], cycles: int):
    A = np.asarray(model["A"], dtype=float)
    c = np.asarray(model["c"], dtype=float)
    C = np.asarray(model["C"], dtype=float)
    d = np.asarray(model["d"], dtype=float)
    z0 = np.asarray(model["coordinates"][:, -1], dtype=float)
    rank = A.shape[0]
    outputs = C.shape[0]
    matrix = np.zeros((rank + outputs + 1, rank + outputs + 1), dtype=float)
    matrix[:rank, :rank] = A
    matrix[:rank, -1] = c
    matrix[rank : rank + outputs, :rank] = C
    matrix[rank : rank + outputs, rank : rank + outputs] = np.eye(outputs)
    matrix[rank : rank + outputs, -1] = d
    matrix[-1, -1] = 1.0
    initial = np.zeros(rank + outputs + 1, dtype=float)
    initial[:rank] = z0
    initial[-1] = 1.0
    propagated = np.linalg.matrix_power(matrix, int(cycles)) @ initial
    return propagated[:rank], propagated[rank : rank + outputs]


def _one_step_prediction(model: dict[str, Any], vector: np.ndarray):
    reference = np.asarray(model["reference"], dtype=float)
    basis = np.asarray(model["basis"], dtype=float)
    z = basis.T @ (np.asarray(vector, dtype=float) - reference)
    z_next = np.asarray(model["A"]) @ z + np.asarray(model["c"])
    vector_next = reference + basis @ z_next
    output = np.asarray(model["C"]) @ z + np.asarray(model["d"])
    return vector_next, output


def propagate_dmd_cycles(
    engine,
    controller,
    waveform,
    temperature_K: float,
    cycles_requested: float,
    *,
    requested_project_cycles: float | None = None,
) -> ProjectivePropagationResult:
    config = dmd_config()
    requested = max(float(cycles_requested), 0.0)
    burst_target = min(int(config["burst_cycles"]), int(math.floor(requested)))
    initial = one_cycle_map(
        engine, controller, waveform, temperature_K
    ).state_start
    if burst_target < 2:
        return ProjectivePropagationResult(
            False, 0.0, 0, 0.0, 0.0, 0.0, {}, initial,
            math.inf, math.inf, False, 0, "insufficient_dmd_horizon"
        )

    geometry_before = geometry_signature(engine)
    stochastic_before = capture_stochastic_state(engine)
    states = [initial]
    cycle_results = []
    ledger_keys: set[str] = set()
    for _index in range(burst_target):
        result = one_cycle_map(
            engine,
            controller,
            waveform,
            temperature_K,
            state=states[-1],
        )
        if result.transition_signature_start != result.transition_signature_end:
            return ProjectivePropagationResult(
                False, 0.0, len(cycle_results), 0.0, 0.0, 0.0, {}, initial,
                math.inf, math.inf, False, 0, "discrete_transition_in_dmd_burst"
            )
        cycle_results.append(result)
        states.append(result.state_end)
        ledger_keys.update(result.ledger_delta_per_cycle)

    keys = sorted(ledger_keys)
    outputs = np.zeros((1 + len(keys), len(cycle_results)), dtype=float)
    for index, result in enumerate(cycle_results):
        outputs[0, index] = max(float(result.hazard_action_per_cycle), 0.0)
        for key_index, key in enumerate(keys, start=1):
            outputs[key_index, index] = float(
                result.ledger_delta_per_cycle.get(key, 0.0)
            )

    try:
        model = _fit_affine_model(states, outputs, config)
    except (ValueError, np.linalg.LinAlgError):
        return ProjectivePropagationResult(
            False, 0.0, len(cycle_results), 0.0, 0.0, 0.0, {}, initial,
            math.inf, math.inf, True, 0, "dmd_fit_failed"
        )
    if float(model["training_error"]) > float(config["training_relative_tolerance"]):
        return ProjectivePropagationResult(
            False, 0.0, len(cycle_results), 0.0, 0.0, 0.0, {}, initial,
            float(model["training_error"]), math.inf, True, 0,
            "dmd_training_error_too_large"
        )

    available = max(int(math.floor(requested)) - len(cycle_results), 0)
    proposed = min(
        available,
        int(math.floor(
            requested_project_cycles
            if requested_project_cycles is not None
            else max(int(config["minimum_project_cycles"]), available)
        )),
    )
    minimum = int(config["minimum_project_cycles"])
    attempts = 0
    last_state_error = math.inf
    last_hazard_error = math.inf
    last_transition_ok = False
    failure_reason = "dmd_validation_failed"

    exact_output = np.sum(outputs, axis=1)
    threshold = max(float(getattr(engine, "hazard_threshold_action", 1.0)), 1.0e-300)
    current_action = max(float(getattr(engine, "hazard_action_current", 0.0)), 0.0)
    remaining_action = max(threshold - current_action, 0.0)

    while proposed >= minimum and attempts < int(config["maximum_attempts"]):
        attempts += 1
        try:
            z_projected, cumulative_output = _propagate_augmented(model, proposed)
        except (ValueError, np.linalg.LinAlgError, OverflowError):
            proposed //= 2
            failure_reason = "dmd_matrix_power_failed"
            continue
        if np.any(~np.isfinite(z_projected)) or np.any(~np.isfinite(cumulative_output)):
            proposed //= 2
            failure_reason = "dmd_nonfinite_projection"
            continue

        raw_vector = np.asarray(model["reference"]) + np.asarray(model["basis"]) @ z_projected
        physical_vector = project_physical_state(states[-1], raw_vector)
        projection_error = _relative_vector_error(raw_vector, physical_vector)
        if projection_error > float(config["physical_projection_relative_tolerance"]):
            proposed //= 2
            failure_reason = "dmd_physical_projection_too_large"
            continue
        projected_state = _snapshot_with_vector(states[-1], physical_vector)

        total_action = float(exact_output[0] + cumulative_output[0])
        predicted_rate = float(
            (_one_step_prediction(model, physical_vector)[1])[0]
        )
        guard_action = float(config["event_guard_cycles"]) * max(
            predicted_rate,
            float(outputs[0, -1]),
            0.0,
        )
        if total_action >= max(remaining_action - guard_action, 0.0):
            proposed //= 2
            failure_reason = "dmd_event_guard"
            continue

        verification = one_cycle_map(
            engine,
            controller,
            waveform,
            temperature_K,
            state=projected_state,
        )
        predicted_next, predicted_output = _one_step_prediction(model, physical_vector)
        predicted_next = project_physical_state(projected_state, predicted_next)
        last_state_error = _relative_vector_error(
            verification.state_end.vector, predicted_next
        )
        last_hazard_error = _relative_scalar_error(
            verification.hazard_action_per_cycle,
            max(float(predicted_output[0]), 0.0),
        )
        ledger_error = 0.0
        for key_index, key in enumerate(keys, start=1):
            ledger_error = max(
                ledger_error,
                _relative_scalar_error(
                    verification.ledger_delta_per_cycle.get(key, 0.0),
                    float(predicted_output[key_index]),
                    1.0,
                ),
            )
        last_transition_ok = bool(
            verification.transition_signature_start
            == cycle_results[-1].transition_signature_end
            == verification.transition_signature_end
        )
        if last_state_error > float(config["state_validation_relative_tolerance"]):
            proposed //= 2
            failure_reason = "dmd_state_validation_failed"
            continue
        if last_hazard_error > float(config["hazard_validation_relative_tolerance"]):
            proposed //= 2
            failure_reason = "dmd_hazard_validation_failed"
            continue
        if ledger_error > float(config["ledger_validation_relative_tolerance"]):
            proposed //= 2
            failure_reason = "dmd_ledger_validation_failed"
            continue
        if not last_transition_ok:
            proposed //= 2
            failure_reason = "dmd_transition_validation_failed"
            continue

        ledger_total = {
            key: float(exact_output[key_index] + cumulative_output[key_index])
            for key_index, key in enumerate(keys, start=1)
        }
        for key in ("engine.N_em", "mpz.emitted_total", "mpz.escaped_total"):
            if key in ledger_total and ledger_total[key] < -1.0e-12:
                proposed //= 2
                failure_reason = "dmd_negative_cumulative_ledger"
                break
        else:
            restore_active_state(engine, projected_state)
            apply_ledger_delta(engine, ledger_total)
            normalized = _update_hazard(engine, total_action)
            total_cycles = float(len(cycle_results) + proposed)
            engine.t = float(getattr(engine, "t", 0.0)) + total_cycles * float(
                waveform.period_s
            )
            if hasattr(engine, "K_prev"):
                engine.K_prev = float(waveform.Kmax)
            stochastic_after = capture_stochastic_state(engine)
            if not _stochastic_preserved_except_clock(
                stochastic_before, stochastic_after
            ):
                raise RuntimeError("DMD propagation changed threshold or RNG state")
            if geometry_signature(engine) != geometry_before:
                raise RuntimeError("DMD propagation changed crack geometry")
            return ProjectivePropagationResult(
                True,
                total_cycles,
                len(cycle_results),
                float(proposed),
                total_action,
                normalized,
                ledger_total,
                projected_state,
                last_state_error,
                last_hazard_error,
                last_transition_ok,
                attempts,
                None,
            )
        continue

    return ProjectivePropagationResult(
        False,
        0.0,
        len(cycle_results),
        0.0,
        0.0,
        0.0,
        {},
        initial,
        last_state_error,
        last_hazard_error,
        last_transition_ok,
        attempts,
        failure_reason,
    )


__all__ = [
    "MODEL_ID",
    "dmd_config",
    "propagate_dmd_cycles",
]
