"""Rate-separated reduced state for the reversible v9.14 v7 cycle map.

The raw accumulated/returned source-slip fields are flux ledgers.  Their
difference is constitutive state because it controls blunting.  This module
therefore exposes ``net_slip_m2`` to a reduced propagator and integrates both
gross fields, and every other cumulative accounting field, from resolved
cycle rates.  Restoring a reduced state reconstructs
``accumulated = returned + net`` exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from typing import Any

import numpy as np


MODEL_ID = "v9.14_v7_net_blunting_rate_separated_state_v1"

ACTIVE_ARRAYS = ("mobile_m2", "retained_m2")
LEDGER_ARRAYS = (
    "accumulated_slip_m2",
    "returned_slip_m2",
    "cumulative_source_activations",
    "cumulative_line_content",
    "cumulative_returned_mobile_per_m",
    "cumulative_escaped_mobile_per_m",
    "cumulative_cancelled_slip_line_content",
)
LEDGER_SCALARS = (
    "effective_plastic_dissipation_J_per_m",
    "external_plastic_work_J_per_m",
    "nonlocal_shielding_work_J_per_m",
    "internal_stress_work_J_per_m",
    "effective_plastic_work_J_per_m",
    "cumulative_transport_channel_time_s",
    "cumulative_reverse_channel_time_s",
    "cumulative_mobile_exposure_m2_s",
    "cumulative_reverse_mobile_exposure_m2_s",
)
MONOTONE_LEDGER_FIELDS = frozenset(LEDGER_ARRAYS + (
    "effective_plastic_dissipation_J_per_m",
    "cumulative_transport_channel_time_s",
    "cumulative_reverse_channel_time_s",
    "cumulative_mobile_exposure_m2_s",
    "cumulative_reverse_mobile_exposure_m2_s",
))


@dataclass(frozen=True)
class Field:
    name: str
    shape: tuple[int, ...]
    start: int
    stop: int


@dataclass(frozen=True)
class ReducedState:
    vector: np.ndarray
    fields: tuple[Field, ...]
    geometry_signature: tuple[Any, ...]


def geometry_signature(state) -> tuple[Any, ...]:
    return (
        int(state.c.n_systems), int(state.c.n_bins),
        tuple(np.shape(state.mobile_m2)), tuple(np.shape(state.retained_m2)),
        float(state.dx), float(state.c.mpz_length_m), float(state.extension_m),
    )


def serialize_active_state(state) -> ReducedState:
    arrays = [(name, np.asarray(getattr(state, name), dtype=float)) for name in ACTIVE_ARRAYS]
    net = np.asarray(state.accumulated_slip_m2, dtype=float) - np.asarray(
        state.returned_slip_m2, dtype=float
    )
    arrays.append(("net_slip_m2", net))
    values: list[np.ndarray] = []
    fields: list[Field] = []
    cursor = 0
    for name, array in arrays:
        if np.any(~np.isfinite(array)) or np.any(array < -1.0e-12):
            raise RuntimeError(f"invalid nonnegative v7 active field: {name}")
        flat = np.maximum(array, 0.0).reshape(-1).copy()
        values.append(flat)
        fields.append(Field(name, tuple(array.shape), cursor, cursor + flat.size))
        cursor += flat.size
    return ReducedState(np.concatenate(values), tuple(fields), geometry_signature(state))


def restore_active_state(state, snapshot: ReducedState, vector=None) -> None:
    if geometry_signature(state) != snapshot.geometry_signature:
        raise RuntimeError("cannot restore v7 reduced state across geometry change")
    data = np.asarray(snapshot.vector if vector is None else vector, dtype=float)
    if data.shape != snapshot.vector.shape or np.any(~np.isfinite(data)):
        raise ValueError("invalid v7 reduced-state vector")
    restored = {}
    for field in snapshot.fields:
        restored[field.name] = np.maximum(
            data[field.start:field.stop].reshape(field.shape), 0.0
        ).copy()
    state.mobile_m2 = restored["mobile_m2"]
    state.retained_m2 = restored["retained_m2"]
    # returned is the gross-return ledger; net is the feedback variable.
    returned = np.maximum(np.asarray(state.returned_slip_m2, dtype=float), 0.0)
    state.returned_slip_m2 = returned
    state.accumulated_slip_m2 = returned + restored["net_slip_m2"]


def capture_ledgers(state) -> dict[str, np.ndarray | float]:
    result: dict[str, np.ndarray | float] = {}
    for name in LEDGER_ARRAYS:
        if hasattr(state, name):
            result[name] = np.asarray(getattr(state, name), dtype=float).copy()
    for name in LEDGER_SCALARS:
        if hasattr(state, name):
            value = float(getattr(state, name))
            result[name] = value if math.isfinite(value) else 0.0
    return result


def ledger_delta(before, after) -> dict[str, np.ndarray | float]:
    result = {}
    for name in set(before) | set(after):
        a = before.get(name, 0.0)
        b = after.get(name, 0.0)
        result[name] = np.asarray(b) - np.asarray(a) if isinstance(b, np.ndarray) else float(b) - float(a)
    return result


def apply_ledger_delta(state, delta, factor: float = 1.0) -> None:
    multiplier = max(float(factor), 0.0)
    for name, increment in delta.items():
        if not hasattr(state, name):
            continue
        current = getattr(state, name)
        value = np.asarray(current) + multiplier * np.asarray(increment)
        if name in MONOTONE_LEDGER_FIELDS:
            value = np.maximum(value, np.asarray(current))
        setattr(state, name, value if isinstance(current, np.ndarray) else float(value))


def independent_cycle(state, loading, controls, cycle_map_fn):
    """Resolve one private authoritative cycle and return separated outputs."""
    trial = copy.deepcopy(state)
    before = capture_ledgers(trial)
    end, hazard, telemetry = cycle_map_fn(trial, loading, controls)
    return end, float(hazard), telemetry, serialize_active_state(end), ledger_delta(before, capture_ledgers(end))


__all__ = [
    "MODEL_ID", "ACTIVE_ARRAYS", "LEDGER_ARRAYS", "LEDGER_SCALARS",
    "MONOTONE_LEDGER_FIELDS", "ReducedState", "apply_ledger_delta",
    "capture_ledgers", "geometry_signature", "independent_cycle",
    "ledger_delta", "restore_active_state", "serialize_active_state",
]
