"""Transactional full-versus-two-half controller for current source updates."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any

from .state_correspondence_r5 import advance_fixed_horizon


MODEL_ID = "r6.transactional-current-source-refresh/1"


@dataclass(frozen=True)
class SourceRefreshTolerance:
    population_relative: float = 1.0e-2
    slip_relative: float = 1.0e-2
    radius_relative: float = 1.0e-2
    rate_relative: float = 5.0e-2
    action_absolute: float = 1.0e-8
    event_time_relative: float = 1.0e-5
    maximum_depth: int = 12


def _observables(engine: Any, parent: Any, duration: float,
                 last_call: dict[str, Any]) -> dict[str, float]:
    mpz = engine.mpz
    diagnostics = mpz.diagnostics()
    gross = float(mpz.cumulative_gross_source_activity)
    returned = float(mpz.cumulative_gross_return_activity)
    parent_gross = float(parent.mpz.cumulative_gross_source_activity)
    parent_returned = float(parent.mpz.cumulative_gross_return_activity)
    return {
        "population": float(mpz.mobile_count + mpz.retained_count),
        "gross_slip": gross,
        "returned_slip": returned,
        "permanent_slip": float(mpz.local_slip_count()),
        "radius_m": float(engine.r_eff()),
        "source_area_m2": float(diagnostics["persistent_site_source_area_m2"]),
        "gross_rate_s": (gross-parent_gross) / max(duration, 1.0e-300),
        "returned_rate_s": (returned-parent_returned) / max(duration, 1.0e-300),
        "hazard_action": float(last_call["hazard_action"]),
    }


def _event_times(run: Any) -> tuple[float, ...]:
    elapsed = 0.0
    times = []
    for row in run.calls:
        elapsed += float(row["consumed_s"])
        if row["fired"]:
            times.append(elapsed)
    return tuple(times)


def _compare(parent: Any, duration: float, full_engine: Any, full_run: Any,
             half_engine: Any, half_runs: tuple,
             tolerance: SourceRefreshTolerance) -> tuple[bool, dict[str, float]]:
    full_values = _observables(full_engine, parent, duration, full_run.calls[-1])
    half_values = _observables(half_engine, parent, duration,
                               half_runs[-1].calls[-1])
    errors: dict[str, float] = {}
    for key in ("population", "gross_slip", "returned_slip", "permanent_slip"):
        a, b = full_values[key], half_values[key]
        # One source packet is not a meaningful relative-error origin.  The
        # absolute floor is a numerical comparison scale only; it does not
        # alter a material threshold or the accepted state.
        errors[key] = abs(a-b) / max(abs(a), abs(b), 100.0)
    for key in ("radius_m", "source_area_m2"):
        a, b = full_values[key], half_values[key]
        errors[key] = abs(a-b) / max(abs(a), abs(b), 1.0e-30)
    for key in ("gross_rate_s", "returned_rate_s"):
        a, b = full_values[key], half_values[key]
        errors[key] = abs(a-b) / max(abs(a), abs(b), 1.0)
    errors["hazard_action_absolute"] = abs(
        full_values["hazard_action"]-half_values["hazard_action"]
    )
    full_times = _event_times(full_run)
    half_times = _event_times(half_runs[0]) + tuple(
        half_runs[0].requested_s + value for value in _event_times(half_runs[1])
    )
    if len(full_times) != len(half_times):
        errors["event_time"] = math.inf
    else:
        errors["event_time"] = max((
            abs(a-b)/max(abs(a), abs(b), 1.0e-15)
            for a, b in zip(full_times, half_times)
        ), default=0.0)
    passed = (
        errors["population"] <= tolerance.population_relative
        and max(errors[key] for key in
                ("gross_slip", "returned_slip", "permanent_slip"))
            <= tolerance.slip_relative
        and max(errors[key] for key in ("radius_m", "source_area_m2"))
            <= tolerance.radius_relative
        and max(errors[key] for key in ("gross_rate_s", "returned_rate_s"))
            <= tolerance.rate_relative
        and errors["hazard_action_absolute"] <= tolerance.action_absolute
        and errors["event_time"] <= tolerance.event_time_relative
    )
    return passed, errors


def advance_current_source_adaptive(
    engine: Any, *, applied_K_Pa_sqrt_m: float, temperature_K: float,
    duration_s: float, tolerance: SourceRefreshTolerance = SourceRefreshTolerance(),
) -> tuple[Any, dict[str, Any]]:
    """Return a privately advanced accepted owner and its error certificate.

    The input owner is never mutated.  Each trial begins from a complete deep
    copy, so rejected paths cannot publish RNG, descriptor, geometry, or source
    state.  The accepted path is always the two-half result.
    """
    if duration_s <= 0.0 or not math.isfinite(duration_s):
        raise ValueError("duration must be positive and finite")
    trials = 0
    accepted = 0
    records: list[dict[str, Any]] = []

    def one(parent: Any, dt: float):
        trial = copy.deepcopy(parent)
        run = advance_fixed_horizon(
            trial, applied_K_Pa_sqrt_m=applied_K_Pa_sqrt_m,
            temperature_K=temperature_K, duration_s=dt,
            maximum_outer_substep_s=dt,
        )
        return trial, run

    def integrate(parent: Any, dt: float, depth: int):
        nonlocal trials, accepted
        full, full_run = one(parent, dt)
        first, first_run = one(parent, 0.5*dt)
        half, second_run = one(first, 0.5*dt)
        trials += 3
        passed, errors = _compare(
            parent, dt, full, full_run, half, (first_run, second_run), tolerance
        )
        records.append({"depth": depth, "duration_s": dt,
                        "accepted": passed, "errors": errors})
        if passed:
            accepted += 2
            return half
        if depth >= tolerance.maximum_depth:
            raise RuntimeError(
                "current source refresh failed prospective convergence: "
                f"duration={dt!r}, errors={errors!r}"
            )
        left = integrate(parent, 0.5*dt, depth+1)
        return integrate(left, 0.5*dt, depth+1)

    result = integrate(engine, float(duration_s), 0)
    return result, {
        "schema": MODEL_ID,
        "requested_duration_s": float(duration_s),
        "controller": "FULL_VERSUS_TWO_HALF_COMPLETE_PRIVATE_STATE",
        "tolerances": tolerance.__dict__,
        "trial_integrations": trials,
        "accepted_subsegments": accepted,
        "maximum_depth_used": max(row["depth"] for row in records),
        "records": records,
        "input_owner_mutated": False,
        "accepted_path": "TWO_HALF",
    }


__all__ = [
    "MODEL_ID", "SourceRefreshTolerance", "advance_current_source_adaptive",
]
