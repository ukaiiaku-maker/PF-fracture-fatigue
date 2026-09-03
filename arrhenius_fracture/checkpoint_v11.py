"""Atomic restart checkpoint for the v11 live-FEM topology transaction state."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
from typing import Any

from .directional_competition_v11 import competition_state_to_dict
from .topology_transaction_v11 import LiveFEMTopologyState, MODEL_ID


SCHEMA = "v11.live-fem-branch-checkpoint/1"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_checkpoint(
    state: LiveFEMTopologyState, path: str | Path, *, provider_runtime: Any = None,
) -> dict[str, Any]:
    """Write the complete accepted state atomically; never checkpoint a trial."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload_object = {"state": state, "provider_runtime": provider_runtime}
    payload = pickle.dumps(payload_object, protocol=5)
    payload_sha = _sha256(payload)
    manifest = {
        "schema": SCHEMA,
        "topology_transaction_model_id": MODEL_ID,
        "state_file": target.name + ".state.pkl",
        "state_sha256": payload_sha,
        "crack_network": state.crack_network.to_dict(),
        "directional_competition": competition_state_to_dict(state.competition),
        "active_tip_ids": list(state.crack_network.active_tip_ids),
        "event_counters": dict(state.event_counters),
        "energy_ledgers": dict(state.energy_ledgers),
        "has_rng_state": state.rng_state is not None,
        "sharp_wake_model_id": state.sharp_wake_model_id,
        "v12_support_state": state.v12_support_state.__dict__ if state.v12_support_state is not None else None,
        "checkpoint_generation": state.checkpoint_generation,
        "provider_runtime": (
            provider_runtime.audit_payload() if provider_runtime is not None else None
        ),
    }
    state_path = target.with_name(manifest["state_file"])
    state_tmp = state_path.with_name(state_path.name + ".tmp")
    manifest_tmp = target.with_name(target.name + ".tmp")
    state_tmp.write_bytes(payload)
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(state_tmp, state_path)
    os.replace(manifest_tmp, target)
    return manifest


def restore_checkpoint(path: str | Path, *, with_provider_runtime: bool = False):
    target = Path(path)
    manifest = json.loads(target.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported v11 live-FEM checkpoint schema")
    state_path = target.with_name(str(manifest["state_file"]))
    payload = state_path.read_bytes()
    if _sha256(payload) != manifest.get("state_sha256"):
        raise ValueError("v11 checkpoint state hash mismatch")
    restored = pickle.loads(payload)
    if isinstance(restored, LiveFEMTopologyState):
        state, provider_runtime = restored, None
    else:
        state = restored.get("state")
        provider_runtime = restored.get("provider_runtime")
    if not isinstance(state, LiveFEMTopologyState):
        raise ValueError("v11 checkpoint payload has the wrong state type")
    if state.crack_network.to_dict() != manifest.get("crack_network"):
        raise ValueError("v11 checkpoint crack network does not match manifest")
    if competition_state_to_dict(state.competition) != manifest.get("directional_competition"):
        raise ValueError("v11 checkpoint directional state does not match manifest")
    if state.sharp_wake_model_id != manifest.get("sharp_wake_model_id", "sharp_wake_causal_v11"):
        raise ValueError("checkpoint sharp-wake model identity mismatch")
    expected_support=state.v12_support_state.__dict__ if state.v12_support_state is not None else None
    if expected_support != manifest.get("v12_support_state"):
        raise ValueError("checkpoint V12 support ownership mismatch")
    return (state, provider_runtime) if with_provider_runtime else state


__all__ = ["SCHEMA", "restore_checkpoint", "write_checkpoint"]
