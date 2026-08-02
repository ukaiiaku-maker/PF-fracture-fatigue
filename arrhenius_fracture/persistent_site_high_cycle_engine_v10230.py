"""Production high-cycle state machine for v10.2.30 fatigue.

Operating modes:

TRANSIENT -> PERIODIC_SEARCH -> STATIONARY_TAIL
                   |                  |
                   v                  v
             SLOW_PROJECTIVE     FIRST_PASSAGE
                   |                  |
                   +--------> EVENT / RESTART

The engine is designed for native horizons through 1e12 cycles. It never uses a
Paris law, never draws extra stochastic thresholds, and never carries an
accelerated representation across a crack-geometry change.
"""
from __future__ import annotations

import math
import os
import time
from typing import Any

from . import persistent_site_forward_robust_v10230 as _transient
from .persistent_site_high_cycle_propagation_v10230 import (
    projective_config,
    propagate_projective_cycles,
    propagate_stationary_cycles,
)
from .persistent_site_high_cycle_state_v10230 import (
    capture_ledgers,
    capture_stochastic_state,
    geometry_signature,
    ledger_delta,
    residual_metrics,
)
from .persistent_site_periodic_solver_v10230 import solve_periodic_state
from .persistent_site_poincare_v10230 import one_cycle_map


MODEL_ID = "v10.2.30_production_high_cycle_state_machine_v1"


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


def high_cycle_config() -> dict[str, float | int]:
    return {
        "stationary_relative_tolerance": _env_float(
            "V10230_HIGH_CYCLE_STATIONARY_REL_TOL", 1.0e-7, 1.0e-14
        ),
        "stationary_diagnostic_tolerance": _env_float(
            "V10230_HIGH_CYCLE_STATIONARY_DIAGNOSTIC_TOL", 1.0e-5, 1.0e-14
        ),
        "stationary_admission_distance": _env_float(
            "V10230_HIGH_CYCLE_STATIONARY_ADMISSION_DISTANCE", 1.0e-6, 1.0e-14
        ),
        "transient_fallback_cycles": _env_float(
            "V10230_HIGH_CYCLE_TRANSIENT_FALLBACK_CYCLES", 64.0, 1.0e-6
        ),
        "maximum_mode_operations": _env_int(
            "V10230_HIGH_CYCLE_MAX_MODE_OPERATIONS", 64, 1
        ),
        "projective_growth_factor": _env_float(
            "V10230_PROJECTIVE_GROWTH_FACTOR", 8.0, 1.0
        ),
        "maximum_projective_cycles": _env_float(
            "V10230_PROJECTIVE_MAX_CYCLES", 1.0e9, 1.0
        ),
        "heartbeat_operations": _env_int(
            "V10230_HIGH_CYCLE_HEARTBEAT_OPERATIONS", 4, 1
        ),
    }


def _sum_numeric(target: dict[str, float], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, (int, float)):
            target[key] = target.get(key, 0.0) + float(value)


def _cache(engine) -> dict[str, Any]:
    signature = geometry_signature(engine)
    cache = getattr(engine, "_v10230_high_cycle_cache", None)
    if not isinstance(cache, dict) or cache.get("geometry_signature") != signature:
        pconfig = projective_config()
        cache = {
            "geometry_signature": signature,
            "projective_next_cycles": max(
                float(pconfig["minimum_project_cycles"]),
                float(pconfig["initial_factor"]) * float(pconfig["burst_cycles"]),
            ),
            "periodic_attempts": 0,
            "last_periodic_converged": False,
        }
        engine._v10230_high_cycle_cache = cache
    return cache


def invalidate_high_cycle_cache(engine, reason: str) -> None:
    pconfig = projective_config()
    engine._v10230_high_cycle_cache = {
        "geometry_signature": geometry_signature(engine),
        "projective_next_cycles": max(
            float(pconfig["minimum_project_cycles"]),
            float(pconfig["initial_factor"]) * float(pconfig["burst_cycles"]),
        ),
        "periodic_attempts": 0,
        "last_periodic_converged": False,
        "invalidated_reason": str(reason),
    }


def _projective_with_requested_scale(
    engine,
    controller,
    waveform,
    temperature_K: float,
    cycles_requested: float,
    requested_project_cycles: float,
):
    burst = max(int(projective_config()["burst_cycles"]), 1)
    factor = max(float(requested_project_cycles) / float(burst), 1.0)
    previous = os.environ.get("V10230_PROJECTIVE_INITIAL_FACTOR")
    os.environ["V10230_PROJECTIVE_INITIAL_FACTOR"] = f"{factor:.17g}"
    try:
        return propagate_projective_cycles(
            engine, controller, waveform, temperature_K, cycles_requested
        )
    finally:
        if previous is None:
            os.environ.pop("V10230_PROJECTIVE_INITIAL_FACTOR", None)
        else:
            os.environ["V10230_PROJECTIVE_INITIAL_FACTOR"] = previous


def _accumulate_transient(
    result: dict[str, Any],
    plastic: dict[str, float],
    advance: dict[str, float],
) -> tuple[float, float, float, float, float, float, int, bool]:
    cycles = max(float(result.get("coupled_hazard_cycles_consumed", 0.0)), 0.0)
    dB = max(float(result.get("dB", 0.0)), 0.0)
    dH = max(float(result.get("physical_hazard_action_step", 0.0)), 0.0)
    da = max(float(result.get("da", 0.0)), 0.0)
    packet_mean = max(float(result.get("packet_mean", 0.0)), 0.0)
    packet_variance = max(float(result.get("packet_variance_m2", 0.0)), 0.0)
    microsteps = int(result.get("microsteps", 0))
    _sum_numeric(plastic, dict(result.get("plastic", {})))
    _sum_numeric(advance, dict(result.get("advance", {})))
    return (
        cycles,
        dB,
        dH,
        da,
        packet_mean,
        packet_variance,
        microsteps,
        bool(result.get("fired", False)),
    )


def integrate_state_coupled_waveform(
    engine,
    controller,
    waveform,
    temperature_K: float,
    cycles_requested: float,
) -> dict[str, Any]:
    requested = max(float(cycles_requested), 0.0)
    period = max(float(waveform.period_s), 0.0)
    config = high_cycle_config()
    if requested <= 0.0 or period <= 0.0:
        return _transient.integrate_state_coupled_waveform(
            engine, controller, waveform, temperature_K, requested
        )
    if getattr(engine, "_energy_gate_pending", None) is not None:
        raise RuntimeError("high-cycle propagation cannot start with a pending crack event")

    cache = _cache(engine)
    initial_geometry = geometry_signature(engine)
    initial_ledgers = capture_ledgers(engine)
    initial_stochastic = capture_stochastic_state(engine)
    consumed = 0.0
    dB_total = 0.0
    dH_total = 0.0
    da_total = 0.0
    packet_mean_total = 0.0
    packet_variance_total = 0.0
    microsteps = 0
    plastic: dict[str, float] = {}
    advance: dict[str, float] = {}
    fired = False
    work_budget_exhausted = False
    modes: list[dict[str, Any]] = []
    operations = 0
    start_wall = time.monotonic()
    last_result: dict[str, Any] = {}

    while consumed < requested and not fired:
        operations += 1
        if operations > int(config["maximum_mode_operations"]):
            work_budget_exhausted = True
            break
        remaining = requested - consumed
        cycle = one_cycle_map(engine, controller, waveform, temperature_K)
        current_residual = residual_metrics(
            cycle.state_start,
            cycle.state_end,
            relative_tolerance=float(config["stationary_relative_tolerance"]),
            diagnostic_tolerance=float(config["stationary_diagnostic_tolerance"]),
        )
        periodic = solve_periodic_state(
            engine,
            controller,
            waveform,
            temperature_K,
            initial_state=cycle.state_start,
        )
        cache["periodic_attempts"] = int(cache.get("periodic_attempts", 0)) + 1
        cache["last_periodic_converged"] = bool(periodic.converged)
        modes.append(
            {
                "mode": "periodic_search",
                "cycles": 0.0,
                "converged": periodic.converged,
                "iterations": periodic.iterations,
                "map_evaluations": periodic.map_evaluations,
                "current_map_residual": current_residual.maximum_relative,
                "fixed_point_residual": periodic.residual.maximum_relative,
                "distance_from_current": periodic.distance_from_initial,
                "failure_reason": periodic.failure_reason,
            }
        )

        stationary_admissible = bool(
            current_residual.converged
            and periodic.converged
            and periodic.distance_from_initial
            <= float(config["stationary_admission_distance"])
        )
        if stationary_admissible:
            stationary = propagate_stationary_cycles(
                engine, waveform, cycle, remaining
            )
            consumed += stationary.cycles_consumed
            dB_total += stationary.normalized_clock_added
            dH_total += stationary.hazard_action_added
            modes.append(
                {
                    "mode": "stationary_tail",
                    "cycles": stationary.cycles_consumed,
                    "event_within_guard": stationary.event_within_guard,
                    "state_residual": current_residual.maximum_relative,
                    "fixed_point_distance": periodic.distance_from_initial,
                    "hazard_action_per_cycle": cycle.hazard_action_per_cycle,
                }
            )
            if consumed >= requested:
                break
            remaining = requested - consumed
            local = _transient.integrate_state_coupled_waveform(
                engine, controller, waveform, temperature_K, remaining
            )
            (
                local_cycles,
                local_dB,
                local_dH,
                local_da,
                local_packet_mean,
                local_packet_variance,
                local_microsteps,
                fired,
            ) = _accumulate_transient(local, plastic, advance)
            consumed += local_cycles
            dB_total += local_dB
            dH_total += local_dH
            da_total += local_da
            packet_mean_total += local_packet_mean
            packet_variance_total += local_packet_variance
            microsteps += local_microsteps
            last_result = dict(local)
            modes.append(
                {
                    "mode": "event_guard_transient",
                    "cycles": local_cycles,
                    "fired": fired,
                }
            )
            if fired:
                invalidate_high_cycle_cache(engine, "first_passage_event")
            if local_cycles <= 0.0 or bool(
                local.get("coupled_hazard_work_budget_exhausted", False)
            ):
                work_budget_exhausted = True
                break
            continue

        requested_project = min(
            float(cache.get("projective_next_cycles", 1.0)),
            float(config["maximum_projective_cycles"]),
            remaining,
        )
        projected = _projective_with_requested_scale(
            engine,
            controller,
            waveform,
            temperature_K,
            remaining,
            requested_project,
        )
        if projected.accepted and projected.cycles_consumed > 0.0:
            consumed += projected.cycles_consumed
            dB_total += projected.normalized_clock_added
            dH_total += projected.hazard_action_added
            cache["projective_next_cycles"] = min(
                max(
                    projected.projected_cycles
                    * float(config["projective_growth_factor"]),
                    requested_project,
                ),
                float(config["maximum_projective_cycles"]),
            )
            modes.append(
                {
                    "mode": "slow_projective",
                    "cycles": projected.cycles_consumed,
                    "burst_cycles": projected.burst_cycles,
                    "projected_cycles": projected.projected_cycles,
                    "drift_error": projected.drift_relative_error,
                    "hazard_error": projected.hazard_relative_error,
                    "attempts": projected.attempts,
                }
            )
            continue

        cache["projective_next_cycles"] = max(
            float(projective_config()["minimum_project_cycles"]),
            0.5 * requested_project,
        )
        fallback_cycles = min(
            remaining, float(config["transient_fallback_cycles"])
        )
        local = _transient.integrate_state_coupled_waveform(
            engine, controller, waveform, temperature_K, fallback_cycles
        )
        (
            local_cycles,
            local_dB,
            local_dH,
            local_da,
            local_packet_mean,
            local_packet_variance,
            local_microsteps,
            fired,
        ) = _accumulate_transient(local, plastic, advance)
        consumed += local_cycles
        dB_total += local_dB
        dH_total += local_dH
        da_total += local_da
        packet_mean_total += local_packet_mean
        packet_variance_total += local_packet_variance
        microsteps += local_microsteps
        last_result = dict(local)
        modes.append(
            {
                "mode": "transient_reference",
                "cycles": local_cycles,
                "fired": fired,
                "projective_failure": projected.failure_reason,
            }
        )
        if fired:
            invalidate_high_cycle_cache(engine, "first_passage_event")
        if local_cycles <= 0.0 or bool(
            local.get("coupled_hazard_work_budget_exhausted", False)
        ):
            work_budget_exhausted = True
            break

        if operations % int(config["heartbeat_operations"]) == 0:
            print(
                "  v10.2.30 high-cycle heartbeat: "
                f"consumed={consumed:.9g}/{requested:.9g} "
                f"mode={modes[-1]['mode']} operations={operations}",
                flush=True,
            )

    if fired:
        final_lambda = max(float(last_result.get("lambda_c", 0.0)), 0.0)
        final_sigma = max(float(last_result.get("sigma_tip", 0.0)), 0.0)
    else:
        final_cycle = one_cycle_map(engine, controller, waveform, temperature_K)
        final_lambda = float(final_cycle.hazard_action_per_cycle / period)
        final_sigma = float(
            final_cycle.state_start.diagnostics.get("sigma_tip_Pa", 0.0)
        )
    final_ledgers = capture_ledgers(engine)
    result = dict(last_result)
    result.update(
        {
            "fired": bool(fired),
            "n_fire": 1 if fired else 0,
            "v_crack": da_total / (consumed * period) if consumed > 0.0 else 0.0,
            "dB": float(dB_total),
            "physical_hazard_action_step": float(dH_total),
            "da": float(da_total),
            "dt_consumed": float(consumed * period),
            "dt_unused": max((requested - consumed) * period, 0.0),
            "packet_mean": float(packet_mean_total),
            "packet_variance_m2": float(packet_variance_total),
            "lambda_c": final_lambda,
            "lambda_c_raw": final_lambda,
            "sigma_tip": final_sigma,
            "plastic": plastic,
            "advance": advance,
            "microsteps": int(microsteps),
            "coupled_hazard_model_id": MODEL_ID,
            "coupled_hazard_high_cycle_engine": True,
            "coupled_hazard_phase_resolved_poincare_map": True,
            "coupled_hazard_periodic_solver": True,
            "coupled_hazard_stationary_first_passage": True,
            "coupled_hazard_slow_projective": True,
            "coupled_hazard_event_restart": True,
            "coupled_hazard_cycles_requested": float(requested),
            "coupled_hazard_cycles_consumed": float(consumed),
            "coupled_hazard_work_budget_exhausted": bool(work_budget_exhausted),
            "coupled_hazard_partial_return": bool(
                not fired and consumed + 1.0e-12 < requested
            ),
            "coupled_hazard_event_localized": bool(fired and consumed < requested),
            "coupled_hazard_mode_operations": int(operations),
            "coupled_hazard_modes": modes,
            "coupled_hazard_wall_seconds": float(time.monotonic() - start_wall),
            "coupled_hazard_geometry_preserved_before_event": bool(
                fired or geometry_signature(engine) == initial_geometry
            ),
            "coupled_hazard_ledger_delta": ledger_delta(
                initial_ledgers, final_ledgers
            ),
            "coupled_hazard_stochastic_threshold_preserved_until_event": bool(
                fired
                or capture_stochastic_state(engine)["hazard_threshold_action"]
                == initial_stochastic["hazard_threshold_action"]
            ),
            "coupled_hazard_config": dict(config),
        }
    )
    return result


__all__ = [
    "MODEL_ID",
    "high_cycle_config",
    "integrate_state_coupled_waveform",
    "invalidate_high_cycle_cache",
]
