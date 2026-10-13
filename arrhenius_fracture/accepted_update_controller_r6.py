"""Transactional full-versus-two-half controller for current source updates."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .reduced_persistent_reference_r2 import advance_signed
from .state_correspondence_r5 import advance_fixed_horizon
from .stochastic_avalanche_tip import pending_geometry_event_scope


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


@dataclass(frozen=True)
class EventAwareAcceptedUpdate:
    engine: Any
    result: dict[str, Any]
    geometry_events: tuple[dict[str, Any], ...]
    consumed_s: float
    audit: dict[str, Any]


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


def _resolved_state(engine: Any, result: dict[str, Any], *, parent: Any,
                    consumed_s: float) -> dict[str, Any]:
    """Mechanically relevant state, retaining signed and bin resolution."""
    arrays = {}
    for name, value in vars(engine.mpz).items():
        if isinstance(value, np.ndarray) and any(token in name for token in (
            "mobile", "retained", "slip", "wake", "source", "backstress",
        )):
            arrays[name] = np.asarray(value, float).copy()
    gross_source = float(engine.mpz.cumulative_gross_source_activity)
    gross_return = float(engine.mpz.cumulative_gross_return_activity)
    parent_source = float(parent.mpz.cumulative_gross_source_activity)
    parent_return = float(parent.mpz.cumulative_gross_return_activity)
    interval = max(float(consumed_s), np.finfo(float).tiny)
    scalars = {
        "gross_source": gross_source,
        "gross_return": gross_return,
        "permanent_slip": float(engine.mpz.local_slip_count()),
        "radius_m": float(engine.r_eff()),
        "hazard_action": float(result["hazard_action_current"]),
        # Native step results do not expose these keys.  The accepted gross
        # ledger increments over the corresponding consumed interval are the
        # actual observable rates; an absent result key is never interpreted
        # as physical zero activity.
        "gross_rate_s": (gross_source-parent_source)/interval,
        "returned_rate_s": (gross_return-parent_return)/interval,
        "sigma_tip_Pa": float(result.get("sigma_tip", 0.0)),
    }
    return {"arrays": arrays, "scalars": scalars}


def advance_current_source_to_earliest_event(
    engine: Any, *, applied_K_Pa_sqrt_m: float, temperature_K: float,
    duration_s: float, tolerance: SourceRefreshTolerance = SourceRefreshTolerance(),
) -> EventAwareAcceptedUpdate:
    """Accepted source update that never consumes beyond its earliest event.

    This is the production-preview counterpart of the fixed-horizon R6
    controller.  Rejected full and split trials own independent random and
    descriptor state.  The accepted split path stops at its first native event;
    it never advances an event remainder that belongs to the coupled scheduler.
    """
    requested = float(duration_s)
    if not math.isfinite(requested) or requested <= 0.0:
        raise ValueError("duration must be positive and finite")
    records: list[dict[str, Any]] = []
    trial_count = 0

    def one(parent: Any, dt: float):
        nonlocal trial_count
        trial_count += 1
        trial = copy.deepcopy(parent)
        with pending_geometry_event_scope() as pending:
            result = advance_signed(
                trial, K_Pa_sqrt_m=float(applied_K_Pa_sqrt_m),
                temperature_K=float(temperature_K), dt_s=float(dt),
            )
            events = tuple(copy.deepcopy(item) for item in pending)
        return trial, result, events, float(result["kinetic_dt_consumed_s"])

    def composed_half(parent: Any, dt: float):
        first = one(parent, 0.5*dt)
        if bool(first[1]["fired"]):
            return first
        second = one(first[0], 0.5*dt)
        return second[0], second[1], first[2]+second[2], 0.5*dt+second[3]

    def compare(parent, full, half, dt: float):
        left = _resolved_state(full[0], full[1], parent=parent, consumed_s=full[3])
        right = _resolved_state(half[0], half[1], parent=parent, consumed_s=half[3])
        errors: dict[str, float] = {}
        array_error = 0.0
        names = sorted(set(left["arrays"]) | set(right["arrays"]))
        for name in names:
            a = left["arrays"].get(name)
            b = right["arrays"].get(name)
            if a is None or b is None or a.shape != b.shape:
                array_error = math.inf
                break
            array_error = max(array_error, float(np.max(np.abs(a-b))) /
                              max(float(np.max(np.abs(a))),
                                  float(np.max(np.abs(b))), 100.0))
        errors["signed_bin_resolved_population_slip"] = array_error
        for key in ("gross_source", "gross_return", "permanent_slip"):
            a, b = left["scalars"][key], right["scalars"][key]
            errors[key] = abs(a-b) / max(abs(a), abs(b), 100.0)
        for key in ("radius_m",):
            a, b = left["scalars"][key], right["scalars"][key]
            errors[key] = abs(a-b) / max(abs(a), abs(b), 1.0e-30)
        for key in ("gross_rate_s", "returned_rate_s", "sigma_tip_Pa"):
            a, b = left["scalars"][key], right["scalars"][key]
            errors[key] = abs(a-b) / max(abs(a), abs(b), 1.0)
        errors["hazard_action_absolute"] = abs(
            left["scalars"]["hazard_action"]-right["scalars"]["hazard_action"])
        full_event = bool(full[1]["fired"])
        half_event = bool(half[1]["fired"])
        same_identity = (
            full_event == half_event
            and tuple(item.get("hazard_event_index") for item in full[2])
            == tuple(item.get("hazard_event_index") for item in half[2])
        )
        event_time_error = (
            abs(full[3]-half[3]) / max(abs(full[3]), abs(half[3]), 1.0e-15)
            if full_event and half_event else (0.0 if same_identity else math.inf)
        )
        errors["event_time"] = event_time_error
        errors["event_identity_mismatch"] = 0.0 if same_identity else 1.0
        passed = (
            array_error <= tolerance.population_relative
            and max(errors[key] for key in
                    ("gross_source", "gross_return", "permanent_slip"))
                <= tolerance.slip_relative
            and errors["radius_m"] <= tolerance.radius_relative
            and max(errors[key] for key in
                    ("gross_rate_s", "returned_rate_s", "sigma_tip_Pa"))
                <= tolerance.rate_relative
            and errors["hazard_action_absolute"] <= tolerance.action_absolute
            and event_time_error <= tolerance.event_time_relative
            and same_identity
        )
        return passed, errors

    def integrate(parent: Any, dt: float, depth: int):
        full = one(parent, dt)
        half = composed_half(parent, dt)
        passed, errors = compare(parent, full, half, dt)
        records.append({"depth": depth, "duration_s": dt,
                        "accepted": passed, "errors": errors})
        if passed:
            return half
        if depth >= tolerance.maximum_depth:
            raise RuntimeError(
                "event-aware current source update failed convergence: "
                f"duration={dt!r}, errors={errors!r}"
            )
        left = integrate(parent, 0.5*dt, depth+1)
        if bool(left[1]["fired"]):
            return left
        right = integrate(left[0], 0.5*dt, depth+1)
        return right[0], right[1], left[2]+right[2], 0.5*dt+right[3]

    accepted = integrate(engine, requested, 0)
    return EventAwareAcceptedUpdate(
        engine=accepted[0], result=accepted[1], geometry_events=accepted[2],
        consumed_s=accepted[3], audit={
            "schema": "r7.event-aware-current-source-update/1",
            "requested_duration_s": requested,
            "consumed_duration_s": accepted[3],
            "event_fired": bool(accepted[1]["fired"]),
            "trial_integrations": trial_count,
            "maximum_depth_used": max(row["depth"] for row in records),
            "records": records,
            "accepted_path": "TWO_HALF_EARLIEST_EVENT",
            "remainder_owner": "CALLING_COUPLED_SCHEDULER",
            "absolute_floors": {
                "population_and_slip": 100.0,
                "rate_and_stress": 1.0,
                "radius_m": 1.0e-30,
            },
            "material_packet_law_claimed": False,
            "input_owner_mutated": False,
        },
    )


__all__ = [
    "MODEL_ID", "SourceRefreshTolerance", "EventAwareAcceptedUpdate",
    "advance_current_source_adaptive", "advance_current_source_to_earliest_event",
]
