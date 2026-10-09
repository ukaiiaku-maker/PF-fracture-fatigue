"""Atomic restart checkpoint for the v11 live-FEM topology transaction state."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import gzip
from typing import Any

from .directional_competition_v11 import competition_state_to_dict
from .topology_transaction_v11 import LiveFEMTopologyState, MODEL_ID
from .vacancy_transport_phase_field_v2 import _state_payload as _transport_state_payload


SCHEMA = "v11.live-fem-branch-checkpoint/1"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_checkpoint(
    state: LiveFEMTopologyState, path: str | Path, *, provider_runtime: Any = None,
    failure_injector=None,
    compression: str | None = None,
) -> dict[str, Any]:
    """Write the complete accepted state atomically; never checkpoint a trial."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload_object = {"state": state, "provider_runtime": provider_runtime}
    payload = pickle.dumps(payload_object, protocol=5)
    if compression not in (None, 'gzip'): raise ValueError('unsupported checkpoint compression')
    if compression == 'gzip': payload = gzip.compress(payload, compresslevel=6, mtime=0)
    payload_sha = _sha256(payload)
    manifest = {
        "schema": SCHEMA,
        "topology_transaction_model_id": MODEL_ID,
        # Immutable payload names make the manifest replacement the one
        # publication point. A failed replacement cannot corrupt a prior
        # checkpoint by overwriting the payload it still references.
        "state_file": target.name + "." + payload_sha + (".state.pkl.gz" if compression == 'gzip' else ".state.pkl"),
        "state_sha256": payload_sha,
        "crack_network": state.crack_network.to_dict(),
        "directional_competition": competition_state_to_dict(state.competition),
        "active_tip_ids": list(state.crack_network.active_tip_ids),
        "event_counters": dict(state.event_counters),
        "energy_ledgers": dict(state.energy_ledgers),
        "equilibrium_observables": (
            state.equilibrium_observables.to_dict()
            if state.equilibrium_observables is not None else None
        ),
        "has_rng_state": state.rng_state is not None,
        "sharp_wake_model_id": state.sharp_wake_model_id,
        "v12_support_state": state.v12_support_state.__dict__ if state.v12_support_state is not None else None,
        "has_production_void_state": state.void_state is not None,
        "checkpoint_generation": state.checkpoint_generation,
        "vacancy_fraction_gp": state.vacancy_fraction_gp.tolist(),
        "vacancy_ledgers": dict(state.vacancy_ledgers),
        "vacancy_temperature_K": state.vacancy_temperature_K,
        "vacancy_transport_state_v2": (
            None if state.vacancy_transport_state_v2 is None
            else _transport_state_payload(state.vacancy_transport_state_v2)
        ),
        "provider_runtime": (
            provider_runtime.audit_payload() if provider_runtime is not None else None
        ),
    }
    if compression == 'gzip': manifest['state_encoding'] = 'pickle+gzip/1'
    if state.void_state is not None:
        from .voiding_v5 import fingerprint as void_fingerprint, serialize as serialize_void_state
        manifest["production_void_state"] = json.loads(serialize_void_state(state.void_state))
        manifest["production_void_state_sha256"] = void_fingerprint(state.void_state)
    state_path = target.with_name(manifest["state_file"])
    state_tmp = state_path.with_name(state_path.name + ".tmp")
    manifest_tmp = target.with_name(target.name + ".tmp")
    state_tmp.write_bytes(payload)
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if failure_injector is not None:
        failure_injector("checkpoint_write",state)
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
    encoding=manifest.get('state_encoding','pickle/1')
    if encoding == 'pickle+gzip/1': payload=gzip.decompress(payload)
    elif encoding != 'pickle/1': raise ValueError('unsupported checkpoint state encoding')
    restored = pickle.loads(payload)
    if isinstance(restored, LiveFEMTopologyState):
        state, provider_runtime = restored, None
    else:
        state = restored.get("state")
        provider_runtime = restored.get("provider_runtime")
    if not isinstance(state, LiveFEMTopologyState):
        raise ValueError("v11 checkpoint payload has the wrong state type")
    # Explicit one-way migration for retained pre-vacancy checkpoints.  Such
    # artifacts own no nonequilibrium history, so the only non-invented state
    # is the declared production default: equilibrium at the stored/default
    # temperature with freshly derived ledgers.
    if getattr(state, "vacancy_fraction_gp", None) is None:
        from dataclasses import replace
        import numpy as np
        from .vacancy_coupling_v1 import (
            equilibrium_vacancy_fraction, initial_vacancy_ledgers,
        )
        temperature = float(getattr(state, "vacancy_temperature_K", 900.0))
        vacancy = np.full(state.mesh.ne, equilibrium_vacancy_fraction(temperature))
        state = replace(
            state, vacancy_fraction_gp=vacancy,
            vacancy_ledgers=initial_vacancy_ledgers(vacancy, state.mesh.area_e),
            vacancy_temperature_K=temperature,
        )
    if state.crack_network.to_dict() != manifest.get("crack_network"):
        raise ValueError("v11 checkpoint crack network does not match manifest")
    if competition_state_to_dict(state.competition) != manifest.get("directional_competition"):
        raise ValueError("v11 checkpoint directional state does not match manifest")
    if state.sharp_wake_model_id != manifest.get("sharp_wake_model_id", "sharp_wake_causal_v11"):
        raise ValueError("checkpoint sharp-wake model identity mismatch")
    if "equilibrium_observables" in manifest:
        restored_observables = getattr(state, "equilibrium_observables", None)
        expected_observables = (
            restored_observables.to_dict() if restored_observables is not None else None
        )
        expected_observables = json.loads(json.dumps(
            expected_observables, sort_keys=True, allow_nan=False,
        ))
        if expected_observables != manifest["equilibrium_observables"]:
            raise ValueError("checkpoint equilibrium observations do not match manifest")
    if "vacancy_fraction_gp" in manifest:
        if state.vacancy_fraction_gp.tolist() != manifest["vacancy_fraction_gp"]:
            raise ValueError("checkpoint vacancy field does not match manifest")
        if dict(state.vacancy_ledgers) != manifest.get("vacancy_ledgers"):
            raise ValueError("checkpoint vacancy ledgers do not match manifest")
        if state.vacancy_temperature_K != manifest.get("vacancy_temperature_K"):
            raise ValueError("checkpoint vacancy temperature does not match manifest")
    if "vacancy_transport_state_v2" in manifest:
        actual = (None if state.vacancy_transport_state_v2 is None
                  else _transport_state_payload(state.vacancy_transport_state_v2))
        actual = json.loads(json.dumps(actual, sort_keys=True, allow_nan=False))
        if actual != manifest["vacancy_transport_state_v2"]:
            raise ValueError("checkpoint V2 vacancy transport state mismatch")
    expected_support=state.v12_support_state.__dict__ if state.v12_support_state is not None else None
    expected_support=json.loads(json.dumps(expected_support,sort_keys=True,allow_nan=False))
    if expected_support != manifest.get("v12_support_state"):
        raise ValueError("checkpoint V12 support ownership mismatch")
    if state.void_state is not None:
        from .voiding_v5 import fingerprint as void_fingerprint
        if void_fingerprint(state.void_state) != manifest.get("production_void_state_sha256"):
            raise ValueError("checkpoint production void state mismatch")
    return (state, provider_runtime) if with_provider_runtime else state


__all__ = ["SCHEMA", "restore_checkpoint", "write_checkpoint"]
