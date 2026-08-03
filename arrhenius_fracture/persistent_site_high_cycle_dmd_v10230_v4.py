"""Positivity-preserving, rate-separated affine-DMD propagation for v10.2.30.

This production propagator addresses the failure modes observed in the real
weak-T 0.75 trajectory:

* only the active constitutive state is represented by the affine cycle map;
* non-negative MPZ fields use a positivity-preserving endpoint projection;
* cumulative ledgers are integrated from independently evaluated local rates
  and never participate in DMD state admission;
* the cleavage clock is integrated from exact start/midpoint/end rates with a
  conservative event guard;
* an accepted local map is reused inside a bounded trust region before a fresh
  exact training burst is requested.

No fracture criterion, stochastic threshold, event-length law, material
parameter, or energy gate is changed.
"""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

import numpy as np

from .persistent_site_high_cycle_checkpoint_v10230 import maybe_write_checkpoint
from .persistent_site_high_cycle_dmd_v10230 import (
    _env_float,
    _env_int,
    _one_step_prediction,
    _relative_scalar_error,
    _relative_vector_error,
    _snapshot_with_vector,
    _stochastic_preserved_except_clock,
    _update_hazard,
    dmd_config as _base_dmd_config,
)
from .persistent_site_high_cycle_dmd_v10230_v2 import (
    _fit_affine_model_stabilized,
)
from .persistent_site_high_cycle_propagation_v10230 import (
    ProjectivePropagationResult,
)
from .persistent_site_high_cycle_state_v10230 import (
    ActiveStateSnapshot,
    apply_ledger_delta,
    capture_stochastic_state,
    geometry_signature,
    restore_active_state,
    serialize_active_state,
)
from .persistent_site_poincare_v10230 import one_cycle_map


MODEL_ID = "v10.2.30_rate_separated_positive_state_affine_dmd_v4"
_MONOTONE_LEDGER_KEYS = frozenset(
    {
        "engine.N_em",
        "mpz.emitted_total",
        "mpz.escaped_total",
    }
)


def dmd_config() -> dict[str, float | int]:
    config = dict(_base_dmd_config())
    config.update(
        {
            "reuse_maximum_segments": _env_int(
                "V10230_DMD_REUSE_MAX_SEGMENTS", 16, 1
            ),
            "reuse_growth_factor": _env_float(
                "V10230_DMD_REUSE_GROWTH_FACTOR", 2.0, 1.0
            ),
        }
    )
    return config


def _positive_state_projection(
    template: ActiveStateSnapshot,
    current_vector: np.ndarray,
    raw_vector: np.ndarray,
) -> np.ndarray:
    """Preserve valid raw predictions and map only invalid MPZ values positive."""
    current = np.asarray(current_vector, dtype=float)
    projected = np.asarray(raw_vector, dtype=float).copy()
    if np.any(~np.isfinite(projected)):
        raise FloatingPointError("projected active state is non-finite")
    for field in template.fields:
        if field.owner != "mpz":
            continue
        sl = slice(field.start, field.stop)
        local = projected[sl]
        invalid = local < 0.0
        if not np.any(invalid):
            continue
        baseline = np.maximum(current[sl], 0.0)
        scale = np.maximum(baseline, 1.0)
        mapped = baseline * np.exp(
            np.clip((local - baseline) / scale, -700.0, 0.0)
        )
        mapped = np.where(baseline > 0.0, mapped, 0.0)
        local[invalid] = mapped[invalid]
        projected[sl] = local
    return projected


def _propagate_from_vector(
    model: dict[str, Any], encoded_vector: np.ndarray, cycles: int
) -> np.ndarray:
    reference = np.asarray(model["reference"], dtype=float)
    basis = np.asarray(model["basis"], dtype=float)
    A = np.asarray(model["A"], dtype=float)
    c = np.asarray(model["c"], dtype=float)
    rank = A.shape[0]
    matrix = np.zeros((rank + 1, rank + 1), dtype=float)
    matrix[:rank, :rank] = A
    matrix[:rank, -1] = c
    matrix[-1, -1] = 1.0
    z0 = basis.T @ (np.asarray(encoded_vector, dtype=float) - reference)
    initial = np.concatenate([z0, np.ones(1, dtype=float)])
    propagated = np.linalg.matrix_power(matrix, int(cycles)) @ initial
    return reference + basis @ propagated[:rank]


def _rate_mapping(result) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in result.ledger_delta_per_cycle.items()
    }


def _positive_rate(key: str, value: float) -> float:
    number = float(value)
    return max(number, 0.0) if key in _MONOTONE_LEDGER_KEYS else number


def _simpson_rate(a: float, b: float, c: float) -> float:
    return (float(a) + 4.0 * float(b) + float(c)) / 6.0


def _integrate_ledgers(
    keys: list[str],
    exact_training: dict[str, float],
    start: dict[str, float],
    midpoint: dict[str, float],
    endpoint: dict[str, float],
    projected_cycles: int,
) -> tuple[dict[str, float], float]:
    total = dict(exact_training)
    maximum_relative_span = 0.0
    for key in keys:
        a = _positive_rate(key, start.get(key, 0.0))
        b = _positive_rate(key, midpoint.get(key, 0.0))
        c = _positive_rate(key, endpoint.get(key, 0.0))
        rate = _simpson_rate(a, b, c)
        if key in _MONOTONE_LEDGER_KEYS:
            rate = max(rate, 0.0)
        total[key] = total.get(key, 0.0) + float(projected_cycles) * rate
        span = max(a, b, c) - min(a, b, c)
        maximum_relative_span = max(
            maximum_relative_span,
            abs(span) / max(abs(a), abs(b), abs(c), 1.0),
        )
    return total, maximum_relative_span


def _sum_training_ledgers(results, keys: list[str]) -> dict[str, float]:
    return {
        key: sum(float(row.ledger_delta_per_cycle.get(key, 0.0)) for row in results)
        for key in keys
    }


def _transition_preserved(*results) -> bool:
    signatures = []
    for result in results:
        if result.transition_signature_start != result.transition_signature_end:
            return False
        signatures.append(result.transition_signature_end)
    return bool(signatures) and all(value == signatures[0] for value in signatures)


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
    initial = serialize_active_state(
        engine, waveform=waveform, temperature_K=temperature_K
    )
    if burst_target < 2:
        return ProjectivePropagationResult(
            False, 0.0, 0, 0.0, 0.0, 0.0, {}, initial,
            math.inf, math.inf, False, 0, "insufficient_dmd_horizon"
        )

    geometry_before = geometry_signature(engine)
    states = [initial]
    training = []
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
                False, 0.0, len(training), 0.0, 0.0, 0.0, {}, initial,
                math.inf, math.inf, False, 0, "discrete_transition_in_dmd_burst"
            )
        training.append(result)
        states.append(result.state_end)
        ledger_keys.update(result.ledger_delta_per_cycle)

    hazard_outputs = np.asarray(
        [[max(float(row.hazard_action_per_cycle), 0.0) for row in training]],
        dtype=float,
    )
    try:
        model = _fit_affine_model_stabilized(states, hazard_outputs, config)
    except (ValueError, np.linalg.LinAlgError):
        return ProjectivePropagationResult(
            False, 0.0, len(training), 0.0, 0.0, 0.0, {}, initial,
            math.inf, math.inf, True, 0, "dmd_fit_failed"
        )
    if float(model["training_error"]) > float(config["training_relative_tolerance"]):
        return ProjectivePropagationResult(
            False, 0.0, len(training), 0.0, 0.0, 0.0, {}, initial,
            float(model["training_error"]), math.inf, True, 0,
            "dmd_training_error_too_large"
        )

    keys = sorted(ledger_keys)
    training_ledgers = _sum_training_ledgers(training, keys)
    exact_training_action = sum(
        max(float(row.hazard_action_per_cycle), 0.0) for row in training
    )
    minimum = int(config["minimum_project_cycles"])
    available = max(int(math.floor(requested)) - burst_target, 0)
    proposal = min(
        available,
        int(math.floor(
            requested_project_cycles
            if requested_project_cycles is not None
            else max(minimum, available)
        )),
    )
    if proposal < minimum:
        return ProjectivePropagationResult(
            False, 0.0, len(training), 0.0, 0.0, 0.0, {}, initial,
            math.inf, math.inf, True, 0, "insufficient_projective_horizon"
        )

    total_cycles = 0.0
    total_projected = 0.0
    total_action = 0.0
    total_normalized = 0.0
    total_ledgers: dict[str, float] = {}
    maximum_state_error = 0.0
    maximum_hazard_error = 0.0
    maximum_ledger_rate_span = 0.0
    attempts = 0
    accepted_segments = 0
    failure_reason: str | None = "dmd_validation_failed"
    current_state = states[-1]
    current_vector = np.asarray(current_state.vector, dtype=float).copy()
    first_segment = True

    while (
        proposal >= minimum
        and total_cycles < requested
        and accepted_segments < int(config["reuse_maximum_segments"])
        and attempts < int(config["maximum_attempts"]) * int(config["reuse_maximum_segments"])
    ):
        remaining = max(int(math.floor(requested - total_cycles)), 0)
        local = min(proposal, remaining)
        if local < minimum:
            break
        attempts += 1
        midpoint_cycles = max(local // 2, 1)
        try:
            midpoint_raw = _propagate_from_vector(model, current_vector, midpoint_cycles)
            endpoint_raw = _propagate_from_vector(model, current_vector, local)
            midpoint_vector = _positive_state_projection(
                current_state, current_vector, midpoint_raw
            )
            endpoint_vector = _positive_state_projection(
                current_state, current_vector, endpoint_raw
            )
        except (ValueError, np.linalg.LinAlgError, OverflowError, FloatingPointError):
            proposal //= 2
            failure_reason = "dmd_nonfinite_projection"
            continue

        midpoint_state = _snapshot_with_vector(current_state, midpoint_vector)
        endpoint_state = _snapshot_with_vector(current_state, endpoint_vector)
        start_exact = one_cycle_map(
            engine, controller, waveform, temperature_K, state=current_state
        )
        midpoint_exact = one_cycle_map(
            engine, controller, waveform, temperature_K, state=midpoint_state
        )
        endpoint_exact = one_cycle_map(
            engine, controller, waveform, temperature_K, state=endpoint_state
        )

        predicted_mid_raw, predicted_mid_output = _one_step_prediction(
            model, midpoint_vector
        )
        predicted_end_raw, predicted_end_output = _one_step_prediction(
            model, endpoint_vector
        )
        try:
            predicted_mid_next = _positive_state_projection(
                midpoint_state, midpoint_vector, predicted_mid_raw
            )
            predicted_end_next = _positive_state_projection(
                endpoint_state, endpoint_vector, predicted_end_raw
            )
        except FloatingPointError:
            proposal //= 2
            failure_reason = "dmd_nonfinite_one_step_prediction"
            continue

        midpoint_state_error = _relative_vector_error(
            midpoint_exact.state_end.vector, predicted_mid_next
        )
        endpoint_state_error = _relative_vector_error(
            endpoint_exact.state_end.vector, predicted_end_next
        )
        state_error = max(midpoint_state_error, endpoint_state_error)
        midpoint_hazard_error = _relative_scalar_error(
            midpoint_exact.hazard_action_per_cycle,
            max(float(predicted_mid_output[0]), 0.0),
        )
        endpoint_hazard_error = _relative_scalar_error(
            endpoint_exact.hazard_action_per_cycle,
            max(float(predicted_end_output[0]), 0.0),
        )
        hazard_error = max(midpoint_hazard_error, endpoint_hazard_error)
        transition_ok = _transition_preserved(
            start_exact, midpoint_exact, endpoint_exact
        )
        if state_error > float(config["state_validation_relative_tolerance"]):
            proposal //= 2
            failure_reason = "dmd_state_validation_failed"
            continue
        if hazard_error > float(config["hazard_validation_relative_tolerance"]):
            proposal //= 2
            failure_reason = "dmd_hazard_validation_failed"
            continue
        if not transition_ok:
            proposal //= 2
            failure_reason = "dmd_transition_validation_failed"
            continue

        start_rate = max(float(start_exact.hazard_action_per_cycle), 0.0)
        midpoint_rate = max(float(midpoint_exact.hazard_action_per_cycle), 0.0)
        endpoint_rate = max(float(endpoint_exact.hazard_action_per_cycle), 0.0)
        projected_action = float(local) * _simpson_rate(
            start_rate, midpoint_rate, endpoint_rate
        )
        upper_action = float(local) * max(start_rate, midpoint_rate, endpoint_rate)
        training_action = exact_training_action if first_segment else 0.0
        threshold = max(float(getattr(engine, "hazard_threshold_action", 1.0)), 1.0e-300)
        current_action = max(float(getattr(engine, "hazard_action_current", 0.0)), 0.0)
        remaining_action = max(threshold - current_action, 0.0)
        guard_action = float(config["event_guard_cycles"]) * max(
            start_rate, midpoint_rate, endpoint_rate
        )
        if training_action + upper_action >= max(remaining_action - guard_action, 0.0):
            proposal //= 2
            failure_reason = "dmd_event_guard"
            continue

        start_ledgers = _rate_mapping(start_exact)
        midpoint_ledgers = _rate_mapping(midpoint_exact)
        endpoint_ledgers = _rate_mapping(endpoint_exact)
        segment_keys = sorted(
            set(keys) | set(start_ledgers) | set(midpoint_ledgers) | set(endpoint_ledgers)
        )
        exact_ledgers = training_ledgers if first_segment else {}
        ledger_total, ledger_rate_span = _integrate_ledgers(
            segment_keys,
            exact_ledgers,
            start_ledgers,
            midpoint_ledgers,
            endpoint_ledgers,
            local,
        )
        keys = segment_keys

        segment_stochastic = capture_stochastic_state(engine)
        restore_active_state(engine, endpoint_state)
        apply_ledger_delta(engine, ledger_total)
        segment_action = training_action + projected_action
        normalized = _update_hazard(engine, segment_action)
        segment_cycles = float(local + (burst_target if first_segment else 0))
        engine.t = float(getattr(engine, "t", 0.0)) + segment_cycles * float(
            waveform.period_s
        )
        if hasattr(engine, "K_prev"):
            engine.K_prev = float(waveform.Kmax)
        if not _stochastic_preserved_except_clock(
            segment_stochastic, capture_stochastic_state(engine)
        ):
            raise RuntimeError("rate-separated DMD changed threshold or RNG state")
        if geometry_signature(engine) != geometry_before:
            raise RuntimeError("rate-separated DMD changed crack geometry")

        accepted_segments += 1
        total_cycles += segment_cycles
        total_projected += float(local)
        total_action += segment_action
        total_normalized += normalized
        for key, value in ledger_total.items():
            total_ledgers[key] = total_ledgers.get(key, 0.0) + float(value)
        maximum_state_error = max(maximum_state_error, state_error)
        maximum_hazard_error = max(maximum_hazard_error, hazard_error)
        maximum_ledger_rate_span = max(
            maximum_ledger_rate_span, ledger_rate_span
        )
        current_state = serialize_active_state(
            engine, waveform=waveform, temperature_K=temperature_K
        )
        current_vector = np.asarray(current_state.vector, dtype=float).copy()
        first_segment = False
        failure_reason = None
        maybe_write_checkpoint(
            engine,
            waveform=waveform,
            temperature_K=temperature_K,
            reason="validated_dmd_segment",
            metadata={
                "model_id": MODEL_ID,
                "accepted_segments": accepted_segments,
                "segment_cycles": segment_cycles,
                "total_cycles_in_call": total_cycles,
                "state_error": state_error,
                "hazard_error": hazard_error,
                "ledger_rate_span": ledger_rate_span,
            },
        )
        remaining = max(int(math.floor(requested - total_cycles)), 0)
        if remaining < minimum:
            break
        proposal = min(
            remaining,
            max(
                minimum,
                int(math.ceil(local * float(config["reuse_growth_factor"]))),
            ),
        )

    completed = total_cycles >= requested - 1.0e-12 * max(requested, 1.0)
    if completed:
        total_cycles = requested
        failure_reason = None
    elif accepted_segments >= int(config["reuse_maximum_segments"]):
        failure_reason = "dmd_reuse_trust_segment_budget_exhausted"

    result = ProjectivePropagationResult(
        accepted=bool(accepted_segments > 0),
        cycles_consumed=float(total_cycles),
        burst_cycles=int(burst_target if accepted_segments > 0 else 0),
        projected_cycles=float(total_projected),
        hazard_action_added=float(total_action),
        normalized_clock_added=float(total_normalized),
        ledger_delta=total_ledgers,
        state_end=current_state if accepted_segments > 0 else initial,
        drift_relative_error=float(maximum_state_error),
        hazard_relative_error=float(maximum_hazard_error),
        transition_preserved=True,
        attempts=int(attempts),
        failure_reason=failure_reason,
    )
    result.rate_separated_outputs = True
    result.positivity_preserving_coordinates = True
    result.reused_local_map = bool(accepted_segments > 1)
    result.accepted_reuse_segments = int(accepted_segments)
    result.maximum_ledger_rate_span = float(maximum_ledger_rate_span)
    result.completed_requested_horizon = bool(completed)
    result.model_id = MODEL_ID
    return result


__all__ = ["MODEL_ID", "dmd_config", "propagate_dmd_cycles"]
