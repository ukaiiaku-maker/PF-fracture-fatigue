"""Per-front constitutive strategy ownership for the R8 spatial integration.

The payload, rather than a module-global switch, is authoritative.  A fresh
engine is always reconstructed through its declared factory before numerical
state is restored; runtime-bound current-law methods are therefore never
unpickled or borrowed from the legacy engine.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np


LEGACY_STRATEGY = "LEGACY_UNIFIED_MPZ_V5"
CURRENT_STRATEGY = "CURRENT_PERSISTENT_REVERSIBLE_R8"
CURRENT_STATE_SCHEMA = "r8.current-persistent-front-state/1"


def strategy_from_payload(payload: Mapping[str, Any] | None) -> str:
    if payload is None:
        return LEGACY_STRATEGY
    return str(payload.get("constitutive_strategy", LEGACY_STRATEGY))


def build_current_engine():
    from .current_coupled_execution_r4 import build_current_r4_engine

    engine = build_current_r4_engine()
    # The reduced campaign may install positive placeholder drives.  Spatial
    # R8 always supplies signed FEM-resolved inputs before every accepted call.
    engine._r2_synthetic_drive_setup = False
    engine._front_constitutive_strategy = CURRENT_STRATEGY
    return engine


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    return copy.deepcopy(value)


def capture_current_engine(engine: Any) -> dict[str, Any]:
    from .persistent_site_high_cycle_state_v10230 import (
        capture_ledgers, capture_stochastic_state, serialize_active_state,
    )

    active = serialize_active_state(engine)
    drive = {}
    for name in (
        "_anisotropic_drive_factors", "_anisotropic_tau_signed_Pa",
        "_anisotropic_drive_reliable", "_anisotropic_drive_serial",
        "_reversible_transport_K_signed_Pa_sqrt_m", "_reversible_tip_radius_m",
    ):
        if hasattr(engine.mpz, name):
            drive[name] = _json_safe(getattr(engine.mpz, name))
    frame = {}
    for name in (
        "_spatial_tip_position_xy_m", "_spatial_tip_tangent_xy",
        "_spatial_crystal_frame_angle_rad", "_spatial_material_wake_origin_xy_m",
    ):
        if hasattr(engine, name):
            frame[name] = _json_safe(getattr(engine, name))
    fields = [asdict(field) for field in active.fields]
    identity = {
        "material_manifest_sha256": getattr(engine, "_r2_material_sha256", None),
        "selection_sha256": getattr(engine, "_r2_selection_sha256", None),
        "signed_kernel_family_sha256": getattr(engine, "_r2_kernel_family_sha256", None),
        "taylor_consumer": getattr(engine, "_r2_taylor_law_id", None),
        "engine_type": type(engine).__qualname__,
    }
    return {
        "schema": CURRENT_STATE_SCHEMA,
        "constitutive_strategy": CURRENT_STRATEGY,
        "initialization_policy": "build_current_r4_engine->spatial_signed_drive",
        "identity": identity,
        "active_vector": np.asarray(active.vector, float).copy(),
        "active_fields": fields,
        "geometry_signature": _json_safe(active.geometry_signature),
        "ledgers": _json_safe(capture_ledgers(engine)),
        "stochastic": _json_safe(capture_stochastic_state(engine)),
        "engine_time_s": float(engine.t),
        "signed_drive": drive,
        "material_frame": frame,
    }


def _set_geometry(engine: Any, signature: list[Any] | tuple[Any, ...]) -> None:
    if len(signature) < 5:
        raise ValueError("current front checkpoint has an incomplete geometry signature")
    engine.n_adv = int(signature[0])
    engine.a_adv = float(signature[1])
    engine.micro_advance_total_m = float(signature[2])
    engine.checkpoint_advance_total_m = float(signature[3])
    engine.mpz.advance_total_m = float(signature[4])


def restore_current_engine(payload: Mapping[str, Any]):
    if payload.get("schema") != CURRENT_STATE_SCHEMA:
        raise ValueError("unsupported current front-engine state")
    engine = build_current_engine()
    expected = capture_current_engine(engine)["identity"]
    if dict(payload.get("identity", {})) != expected:
        raise ValueError("current front material/kernel/Taylor identity differs from checkpoint")
    _set_geometry(engine, payload["geometry_signature"])

    from .persistent_site_high_cycle_state_v10230 import (
        ActiveStateSnapshot, StateField,
        restore_active_state, serialize_active_state,
    )
    fields = tuple(StateField(
        owner=str(row["owner"]), name=str(row["name"]),
        shape=tuple(row["shape"]), start=int(row["start"]),
        stop=int(row["stop"]), floor=float(row["floor"]),
    ) for row in payload["active_fields"])
    current = serialize_active_state(engine)
    recorded = payload["geometry_signature"]
    if (int(recorded[5]) != int(current.geometry_signature[5])
            or int(recorded[6]) != int(current.geometry_signature[6])
            or tuple(recorded[7]) != tuple(current.geometry_signature[7])):
        raise ValueError("current front checkpoint discretization differs from factory")
    snapshot = ActiveStateSnapshot(
        vector=np.asarray(payload["active_vector"], float), fields=fields,
        diagnostics=current.diagnostics,
        geometry_signature=current.geometry_signature,
    )
    restore_active_state(engine, snapshot)
    target = {str(key): float(value) for key, value in payload.get("ledgers", {}).items()}
    # This in-state serializer owns absolute checkpoint values.  Assign them
    # directly rather than restoring through floating-point ledger deltas;
    # repeated topology remap/capture cycles must remain bitwise idempotent.
    for key, value in target.items():
        owner_name, name = key.split(".", 1)
        owner = engine if owner_name == "engine" else engine.mpz
        if name.endswith("]") and "[" in name:
            attribute, raw_index = name[:-1].split("[", 1)
            index = tuple(int(item) for item in raw_index.split(","))
            array = np.asarray(getattr(owner, attribute), float).copy()
            array[index] = value
            setattr(owner, attribute, array)
        else:
            setattr(owner, name, value)
    stochastic = payload.get("stochastic", {})
    for name, value in stochastic.items():
        if name == "rng_state":
            continue
        if hasattr(engine, name):
            restored = copy.deepcopy(value)
            if name in {"hazard_threshold_history", "avalanche_event_length_history"}:
                restored = list(restored)
            setattr(engine, name, restored)
    if stochastic.get("rng_state") is not None:
        engine._hazard_rng.bit_generator.state = copy.deepcopy(stochastic["rng_state"])
    engine.t = float(payload.get("engine_time_s", 0.0))
    for name, value in payload.get("signed_drive", {}).items():
        setattr(engine.mpz, name, np.asarray(value, float).copy()
                if isinstance(value, list) else copy.deepcopy(value))
    for name, value in payload.get("material_frame", {}).items():
        setattr(engine, name, tuple(value) if isinstance(value, list) else copy.deepcopy(value))
    engine._front_constitutive_strategy = CURRENT_STRATEGY
    return engine


def payload_fingerprint(payload: Mapping[str, Any]) -> str:
    def normalize(value: Any):
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            return {"dtype": array.dtype.str, "shape": list(array.shape),
                    "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
        return _json_safe(value)
    return hashlib.sha256(json.dumps(
        normalize(payload), sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


__all__ = [
    "CURRENT_STATE_SCHEMA", "CURRENT_STRATEGY", "LEGACY_STRATEGY",
    "build_current_engine", "capture_current_engine", "payload_fingerprint",
    "restore_current_engine", "strategy_from_payload",
]
