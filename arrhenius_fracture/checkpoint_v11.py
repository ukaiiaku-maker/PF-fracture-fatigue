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


def write_checkpoint(state: LiveFEMTopologyState, path: str | Path) -> dict[str, Any]:
    """Write the complete accepted state atomically; never checkpoint a trial."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = pickle.dumps(state, protocol=5)
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
    }
    state_path = target.with_name(manifest["state_file"])
    state_tmp = state_path.with_name(state_path.name + ".tmp")
    manifest_tmp = target.with_name(target.name + ".tmp")
    state_tmp.write_bytes(payload)
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(state_tmp, state_path)
    os.replace(manifest_tmp, target)
    return manifest


def restore_checkpoint(path: str | Path) -> LiveFEMTopologyState:
    target = Path(path)
    manifest = json.loads(target.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported v11 live-FEM checkpoint schema")
    state_path = target.with_name(str(manifest["state_file"]))
    payload = state_path.read_bytes()
    if _sha256(payload) != manifest.get("state_sha256"):
        raise ValueError("v11 checkpoint state hash mismatch")
    state = pickle.loads(payload)
    if not isinstance(state, LiveFEMTopologyState):
        raise ValueError("v11 checkpoint payload has the wrong state type")
    if state.crack_network.to_dict() != manifest.get("crack_network"):
        raise ValueError("v11 checkpoint crack network does not match manifest")
    if competition_state_to_dict(state.competition) != manifest.get("directional_competition"):
        raise ValueError("v11 checkpoint directional state does not match manifest")
    return state


__all__ = ["SCHEMA", "restore_checkpoint", "write_checkpoint"]
