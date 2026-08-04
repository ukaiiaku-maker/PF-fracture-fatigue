"""Atomic live checkpoints for v10.2.30 high-cycle fatigue propagation.

The checkpoint is diagnostic and restart-oriented.  It never changes physics.
A checkpoint contains the complete active MPZ state, cumulative ledgers,
cleavage-clock state, RNG state, geometry signature, and high-cycle cache.
Restoration fails closed if the initialized engine geometry is incompatible.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

from .persistent_site_high_cycle_state_v10230 import (
    apply_ledger_delta,
    capture_ledgers,
    capture_stochastic_state,
    geometry_signature,
    restore_active_state,
    serialize_active_state,
)


MODEL_ID = "v10.2.30_atomic_live_high_cycle_checkpoint_v1"
_OUTER_STATE_PROVIDER = None


def register_outer_state_provider(provider) -> None:
    """Register the active 2-D driver's committed-state snapshot callback."""
    global _OUTER_STATE_PROVIDER
    _OUTER_STATE_PROVIDER = provider


def clear_outer_state_provider() -> None:
    global _OUTER_STATE_PROVIDER
    _OUTER_STATE_PROVIDER = None


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return repr(value)


def _root() -> Path | None:
    raw = os.environ.get("V10230_HIGH_CYCLE_CHECKPOINT_DIR", "").strip()
    return Path(raw).resolve() if raw else None


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".npz", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _field_manifest(snapshot) -> list[dict[str, Any]]:
    return [
        {
            "owner": field.owner,
            "name": field.name,
            "shape": list(field.shape),
            "start": int(field.start),
            "stop": int(field.stop),
            "floor": float(field.floor),
        }
        for field in snapshot.fields
    ]


def write_checkpoint(
    engine,
    *,
    waveform=None,
    temperature_K: float | None = None,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    root = _root()
    if root is None:
        return None
    root.mkdir(parents=True, exist_ok=True)
    snapshot = serialize_active_state(
        engine, waveform=waveform, temperature_K=temperature_K
    )
    period = float(getattr(waveform, "period_s", 0.0) or 0.0)
    cycles = (
        float(getattr(engine, "t", 0.0)) / period
        if period > 0.0
        else None
    )
    payload = {
        "schema": MODEL_ID,
        "timestamp_unix_s": time.time(),
        "reason": str(reason),
        "temperature_K": None if temperature_K is None else float(temperature_K),
        "cycles_from_engine_time": cycles,
        "engine_time_s": float(getattr(engine, "t", 0.0)),
        "geometry_signature": _json_safe(geometry_signature(engine)),
        "diagnostics": _json_safe(snapshot.diagnostics),
        "fields": _field_manifest(snapshot),
        "ledgers": _json_safe(capture_ledgers(engine)),
        "stochastic": _json_safe(capture_stochastic_state(engine)),
        "high_cycle_cache": _json_safe(
            copy.deepcopy(getattr(engine, "_v10230_high_cycle_cache", {}))
        ),
        "metadata": _json_safe(metadata or {}),
    }
    _atomic_npz(root / "high_cycle_live_state.npz", active_vector=snapshot.vector)
    _atomic_text(
        root / "high_cycle_live_checkpoint.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    if _OUTER_STATE_PROVIDER is not None:
        from .run_state_checkpoint_v10230 import write_combined_checkpoint

        outer, outer_arrays = _OUTER_STATE_PROVIDER(engine, payload)
        write_combined_checkpoint(
            root,
            outer=outer,
            arrays=outer_arrays,
            kinetic=payload,
            kinetic_vector=snapshot.vector,
        )
    history_path = root / "high_cycle_live_history.jsonl"
    with history_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
    return payload


def maybe_write_checkpoint(
    engine,
    *,
    waveform=None,
    temperature_K: float | None = None,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    root = _root()
    if root is None:
        return None
    now = time.monotonic()
    minimum_seconds = float(
        os.environ.get("V10230_HIGH_CYCLE_CHECKPOINT_MIN_SECONDS", "30")
    )
    last = float(getattr(engine, "_v10230_checkpoint_wall", -math.inf))
    force = reason in {
        "validated_dmd_segment",
        "first_passage",
        "post_event_restart",
        "integrator_return",
    }
    if not force and now - last < max(minimum_seconds, 0.0):
        return None
    payload = write_checkpoint(
        engine,
        waveform=waveform,
        temperature_K=temperature_K,
        reason=reason,
        metadata=metadata,
    )
    engine._v10230_checkpoint_wall = now
    return payload


def restore_checkpoint(engine, root: str | Path) -> dict[str, Any]:
    directory = Path(root)
    payload = json.loads((directory / "high_cycle_live_checkpoint.json").read_text())
    arrays = np.load(directory / "high_cycle_live_state.npz")
    restore_checkpoint_payload(engine, payload, arrays["active_vector"])
    return payload


def restore_checkpoint_payload(engine, payload: dict[str, Any], vector) -> None:
    """Restore an already validated kinetic payload after outer geometry."""
    current = serialize_active_state(engine)
    recorded_geometry = tuple(payload.get("geometry_signature", []))
    if tuple(_json_safe(geometry_signature(engine))) != recorded_geometry:
        raise RuntimeError(
            "checkpoint geometry differs from the initialized engine; "
            "restore the matching outer crack geometry first"
        )
    vector = np.asarray(vector, dtype=float)
    restore_active_state(engine, current, vector)

    current_ledgers = capture_ledgers(engine)
    target_ledgers = {
        str(key): float(value) for key, value in payload.get("ledgers", {}).items()
    }
    apply_ledger_delta(
        engine,
        {
            key: target_ledgers.get(key, 0.0) - current_ledgers.get(key, 0.0)
            for key in set(current_ledgers) | set(target_ledgers)
        },
    )
    stochastic = payload.get("stochastic", {})
    for name in (
        "B",
        "hazard_threshold_action",
        "hazard_action_current",
        "hazard_event_index",
        "hazard_threshold_history",
        "avalanche_base_checkpoint_m",
        "avalanche_event_advance_m",
        "avalanche_event_length_factor",
        "avalanche_last_completed_advance_m",
        "avalanche_last_completed_factor",
        "avalanche_event_length_history",
        "avalanche_checkpoint_synchronized",
    ):
        if name in stochastic:
            setattr(engine, name, copy.deepcopy(stochastic[name]))
    rng_state = stochastic.get("rng_state")
    if rng_state is not None and getattr(engine, "_hazard_rng", None) is not None:
        engine._hazard_rng.bit_generator.state = copy.deepcopy(rng_state)
    engine.t = float(payload.get("engine_time_s", getattr(engine, "t", 0.0)))
    engine._v10230_high_cycle_cache = copy.deepcopy(
        payload.get("high_cycle_cache", {})
    )
    if hasattr(engine, "avalanche_cfg"):
        from .stochastic_avalanche_tip import threshold_event_length_factor

        expected_factor = threshold_event_length_factor(
            engine.hazard_threshold_action,
            mode=engine.avalanche_cfg.mode,
            minimum_factor=engine.avalanche_cfg.minimum_factor,
            maximum_factor=engine.avalanche_cfg.maximum_factor,
            deterministic_threshold=getattr(engine.hazard_cfg, "mode", "") == "deterministic",
        )
        if not math.isclose(
            float(engine.avalanche_event_length_factor), expected_factor,
            rel_tol=1.0e-14, abs_tol=1.0e-15,
        ):
            raise RuntimeError("restored event-length factor differs from restored threshold")
        expected_advance = float(engine.avalanche_base_checkpoint_m) * expected_factor
        if not math.isclose(
            float(engine.avalanche_event_advance_m), expected_advance,
            rel_tol=1.0e-14, abs_tol=1.0e-18,
        ):
            raise RuntimeError("restored event proposal differs from threshold and da_phys")


__all__ = [
    "MODEL_ID",
    "clear_outer_state_provider",
    "maybe_write_checkpoint",
    "register_outer_state_provider",
    "restore_checkpoint",
    "restore_checkpoint_payload",
    "write_checkpoint",
]
