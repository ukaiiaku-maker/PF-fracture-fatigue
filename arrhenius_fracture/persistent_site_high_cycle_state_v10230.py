"""Complete active-state serialization for the v10.2.30 high-cycle engine.

The high-cycle solver distinguishes four channels that must not be conflated:

* active constitutive state: may approach a periodic fixed point or slow manifold;
* cumulative ledgers: exact accounting quantities, never fixed-point residuals;
* stochastic first-passage state: threshold/action/RNG, never altered by private solves;
* geometry/event state: invalidates every accelerated representation after a crack event.

Only the active constitutive state is vectorized for periodic and projective solves.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from typing import Any, Iterable

import numpy as np


MODEL_ID = "v10.2.30_complete_high_cycle_active_state_v1"

MPZ_ACTIVE_ARRAYS = (
    "mobile_positive",
    "mobile_negative",
    "retained_positive",
    "retained_negative",
    "accumulated_slip_positive",
    "accumulated_slip_negative",
    "wake_mobile_positive",
    "wake_mobile_negative",
    "wake_retained_positive",
    "wake_retained_negative",
    "wake_slip_positive",
    "wake_slip_negative",
)

ENGINE_ACTIVE_SCALARS = (
    "W_emit",
    "K_prev",
)

MPZ_LEDGER_SCALARS = (
    "emitted_total",
    "escaped_total",
    "signed_source_activations_total",
    "signed_line_content_emitted_total",
)

ENGINE_LEDGER_SCALARS = (
    "N_em",
)


@dataclass(frozen=True)
class StateField:
    owner: str
    name: str
    shape: tuple[int, ...]
    start: int
    stop: int
    floor: float


@dataclass(frozen=True)
class ActiveStateSnapshot:
    vector: np.ndarray
    fields: tuple[StateField, ...]
    diagnostics: dict[str, float]
    geometry_signature: tuple[Any, ...]

    def copy(self) -> "ActiveStateSnapshot":
        return ActiveStateSnapshot(
            vector=np.asarray(self.vector, dtype=float).copy(),
            fields=self.fields,
            diagnostics=dict(self.diagnostics),
            geometry_signature=tuple(self.geometry_signature),
        )


@dataclass(frozen=True)
class ResidualMetrics:
    maximum_relative: float
    rms_relative: float
    field_relative: dict[str, float]
    diagnostic_relative: dict[str, float]
    converged: bool


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _owner(engine, name: str):
    if name == "engine":
        return engine
    if name == "mpz":
        return engine.mpz
    raise KeyError(name)


def _sync_derived(mpz) -> None:
    if all(hasattr(mpz, name) for name in ("mobile_positive", "mobile_negative")):
        mpz.mobile = np.asarray(mpz.mobile_positive) + np.asarray(mpz.mobile_negative)
    if all(hasattr(mpz, name) for name in ("retained_positive", "retained_negative")):
        mpz.retained = np.asarray(mpz.retained_positive) + np.asarray(mpz.retained_negative)
    if all(
        hasattr(mpz, name)
        for name in ("accumulated_slip_positive", "accumulated_slip_negative")
    ):
        mpz.accumulated_slip = (
            np.asarray(mpz.accumulated_slip_positive)
            + np.asarray(mpz.accumulated_slip_negative)
        )
    if all(hasattr(mpz, name) for name in ("wake_mobile_positive", "wake_mobile_negative")):
        mpz.wake_mobile = (
            np.asarray(mpz.wake_mobile_positive) + np.asarray(mpz.wake_mobile_negative)
        )
    if all(
        hasattr(mpz, name)
        for name in ("wake_retained_positive", "wake_retained_negative")
    ):
        mpz.wake_retained = (
            np.asarray(mpz.wake_retained_positive)
            + np.asarray(mpz.wake_retained_negative)
        )
    if all(hasattr(mpz, name) for name in ("wake_slip_positive", "wake_slip_negative")):
        mpz.wake_slip = (
            np.asarray(mpz.wake_slip_positive) + np.asarray(mpz.wake_slip_negative)
        )


def geometry_signature(engine) -> tuple[Any, ...]:
    mpz = getattr(engine, "mpz", None)
    return (
        int(getattr(engine, "n_adv", 0)),
        _finite(getattr(engine, "a_adv", 0.0)),
        _finite(getattr(engine, "micro_advance_total_m", 0.0)),
        _finite(getattr(engine, "checkpoint_advance_total_m", 0.0)),
        _finite(getattr(mpz, "advance_total_m", 0.0)),
        int(getattr(mpz, "n_bins", 0)),
        int(getattr(mpz, "wake_n_bins", 0)),
        tuple(np.shape(getattr(mpz, "mobile_positive", np.empty(0)))),
    )


def _diagnostics(engine, waveform=None, temperature_K: float | None = None) -> dict[str, float]:
    mpz = getattr(engine, "mpz", None)
    result = {
        "mobile_count": _finite(getattr(mpz, "mobile_count", 0.0)),
        "retained_count": _finite(getattr(mpz, "retained_count", 0.0)),
        "sigma_back_Pa": _finite(
            getattr(mpz, "continuum_source_last_sigma_back_Pa", 0.0)
        ),
        "emission_hazard_s": max(
            _finite(getattr(mpz, "continuum_source_last_aggregate_hazard_s", 0.0)),
            0.0,
        ),
        "tip_radius_m": max(_finite(engine.r_eff() if hasattr(engine, "r_eff") else 0.0), 0.0),
        "active_K_shield_Pa_sqrt_m": _finite(
            engine.K_shield() if hasattr(engine, "K_shield") else 0.0
        ),
    }
    if waveform is not None:
        sigma = max(_finite(engine.sigma_tip(float(waveform.Kmax))), 0.0)
        result["sigma_tip_Pa"] = sigma
        if temperature_K is not None and hasattr(engine, "lambda_cleave"):
            lam, _raw, _barrier = engine.lambda_cleave(sigma, float(temperature_K))
            result["lambda_cleave_s"] = max(_finite(lam), 0.0)
    return result


def serialize_active_state(
    engine,
    *,
    waveform=None,
    temperature_K: float | None = None,
) -> ActiveStateSnapshot:
    values: list[np.ndarray] = []
    fields: list[StateField] = []
    cursor = 0
    mpz = engine.mpz

    for name in MPZ_ACTIVE_ARRAYS:
        if not hasattr(mpz, name):
            continue
        array = np.asarray(getattr(mpz, name), dtype=float)
        flat = array.reshape(-1).copy()
        values.append(flat)
        stop = cursor + flat.size
        fields.append(StateField("mpz", name, tuple(array.shape), cursor, stop, 1.0))
        cursor = stop

    for name in ENGINE_ACTIVE_SCALARS:
        if not hasattr(engine, name):
            continue
        flat = np.asarray([_finite(getattr(engine, name))], dtype=float)
        values.append(flat)
        fields.append(StateField("engine", name, (), cursor, cursor + 1, 1.0))
        cursor += 1

    vector = np.concatenate(values) if values else np.zeros(0, dtype=float)
    if np.any(~np.isfinite(vector)):
        raise RuntimeError("active-state serialization produced non-finite values")
    return ActiveStateSnapshot(
        vector=vector,
        fields=tuple(fields),
        diagnostics=_diagnostics(engine, waveform, temperature_K),
        geometry_signature=geometry_signature(engine),
    )


def project_physical_state(snapshot: ActiveStateSnapshot, vector: np.ndarray) -> np.ndarray:
    projected = np.asarray(vector, dtype=float).copy()
    if projected.shape != snapshot.vector.shape:
        raise ValueError("active-state vector shape changed")
    if np.any(~np.isfinite(projected)):
        raise ValueError("active-state vector contains non-finite values")
    for field in snapshot.fields:
        if field.owner == "mpz":
            projected[field.start:field.stop] = np.maximum(
                projected[field.start:field.stop], 0.0
            )
    return projected


def restore_active_state(
    engine,
    snapshot: ActiveStateSnapshot,
    vector: np.ndarray | None = None,
) -> None:
    if geometry_signature(engine) != snapshot.geometry_signature:
        raise RuntimeError("cannot restore an active state across a geometry change")
    data = project_physical_state(snapshot, snapshot.vector if vector is None else vector)
    for field in snapshot.fields:
        owner = _owner(engine, field.owner)
        raw = data[field.start:field.stop]
        if field.shape:
            value: Any = raw.reshape(field.shape).copy()
        else:
            value = float(raw[0])
        setattr(owner, field.name, value)
    _sync_derived(engine.mpz)


def capture_ledgers(engine) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in ENGINE_LEDGER_SCALARS:
        if hasattr(engine, name):
            result[f"engine.{name}"] = _finite(getattr(engine, name))
    mpz = engine.mpz
    for name in MPZ_LEDGER_SCALARS:
        if hasattr(mpz, name):
            result[f"mpz.{name}"] = _finite(getattr(mpz, name))
    return result


def ledger_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    keys = set(before) | set(after)
    return {key: _finite(after.get(key, 0.0)) - _finite(before.get(key, 0.0)) for key in keys}


def apply_ledger_delta(engine, delta: dict[str, float], cycles: float = 1.0) -> None:
    factor = max(_finite(cycles), 0.0)
    for key, increment in delta.items():
        owner_name, name = key.split(".", 1)
        owner = _owner(engine, owner_name)
        if not hasattr(owner, name):
            continue
        setattr(owner, name, _finite(getattr(owner, name)) + factor * _finite(increment))


def capture_stochastic_state(engine) -> dict[str, Any]:
    return {
        "B": _finite(getattr(engine, "B", 0.0)),
        "hazard_threshold_action": _finite(
            getattr(engine, "hazard_threshold_action", 1.0), 1.0
        ),
        "hazard_action_current": _finite(getattr(engine, "hazard_action_current", 0.0)),
        "hazard_event_index": int(getattr(engine, "hazard_event_index", 0)),
        "avalanche_base_checkpoint_m": _finite(
            getattr(engine, "avalanche_base_checkpoint_m", 0.0)
        ),
        "avalanche_event_advance_m": _finite(
            getattr(engine, "avalanche_event_advance_m", 0.0)
        ),
        "avalanche_event_length_factor": _finite(
            getattr(engine, "avalanche_event_length_factor", 0.0)
        ),
        "avalanche_last_completed_advance_m": _finite(
            getattr(engine, "avalanche_last_completed_advance_m", 0.0)
        ),
        "avalanche_last_completed_factor": _finite(
            getattr(engine, "avalanche_last_completed_factor", 0.0)
        ),
        "avalanche_event_length_history": tuple(
            float(value) for value in getattr(engine, "avalanche_event_length_history", [])
        ),
        "avalanche_checkpoint_synchronized": bool(
            getattr(engine, "avalanche_checkpoint_synchronized", False)
        ),
        "hazard_threshold_history": tuple(
            float(value) for value in getattr(engine, "hazard_threshold_history", [])
        ),
        "rng_state": copy.deepcopy(
            getattr(getattr(engine, "_hazard_rng", None), "bit_generator", None).state
            if getattr(engine, "_hazard_rng", None) is not None
            else None
        ),
    }


def stochastic_state_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a == b


def residual_metrics(
    reference: ActiveStateSnapshot,
    candidate: ActiveStateSnapshot,
    *,
    relative_tolerance: float = 1.0e-8,
    diagnostic_tolerance: float | None = None,
) -> ResidualMetrics:
    if reference.fields != candidate.fields:
        raise ValueError("active-state layouts differ")
    if reference.geometry_signature != candidate.geometry_signature:
        raise ValueError("active-state residual crossed a geometry change")
    field_relative: dict[str, float] = {}
    weighted: list[float] = []
    for field in reference.fields:
        a = reference.vector[field.start:field.stop]
        b = candidate.vector[field.start:field.stop]
        scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), field.floor)
        value = float(np.linalg.norm(b - a)) / scale
        field_relative[f"{field.owner}.{field.name}"] = value
        weighted.append(value * value)

    diagnostic_relative: dict[str, float] = {}
    for name in sorted(set(reference.diagnostics) & set(candidate.diagnostics)):
        a = _finite(reference.diagnostics[name])
        b = _finite(candidate.diagnostics[name])
        if name.endswith("_s") and a > 0.0 and b > 0.0:
            value = abs(math.log10(b / a))
        else:
            floor = 1.0e-30 if name.endswith("_m") else 1.0
            value = abs(b - a) / max(abs(a), abs(b), floor)
        diagnostic_relative[name] = value

    maximum = max(
        [0.0, *field_relative.values(), *diagnostic_relative.values()]
    )
    rms = math.sqrt(sum(weighted) / max(len(weighted), 1))
    diag_tol = float(relative_tolerance if diagnostic_tolerance is None else diagnostic_tolerance)
    converged = (
        max(field_relative.values(), default=0.0) <= float(relative_tolerance)
        and max(diagnostic_relative.values(), default=0.0) <= diag_tol
    )
    return ResidualMetrics(maximum, rms, field_relative, diagnostic_relative, converged)


def state_distance(reference: ActiveStateSnapshot, candidate: ActiveStateSnapshot) -> float:
    return residual_metrics(reference, candidate, relative_tolerance=math.inf).maximum_relative


__all__ = [
    "MODEL_ID",
    "ActiveStateSnapshot",
    "ResidualMetrics",
    "StateField",
    "apply_ledger_delta",
    "capture_ledgers",
    "capture_stochastic_state",
    "geometry_signature",
    "ledger_delta",
    "project_physical_state",
    "residual_metrics",
    "restore_active_state",
    "serialize_active_state",
    "state_distance",
    "stochastic_state_equal",
]
