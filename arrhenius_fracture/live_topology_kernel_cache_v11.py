"""Persistent exact-state cache for accepted v11 live-topology FEM results."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
from typing import Any, Callable

from .live_topology_kernel_v11 import PROVIDER_ID, SCHEMA


CACHE_SCHEMA = "v11_exact_topology_live_fem_cache_v1"


def cache_key(mechanical_configuration_fingerprint: str, topology_fingerprint: str) -> str:
    token = f"{mechanical_configuration_fingerprint}|{topology_fingerprint}".encode()
    return hashlib.sha256(token).hexdigest()


class ExactTopologyCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _paths(self, key: str) -> tuple[Path, Path]:
        directory = self.root / key
        return directory / "manifest.json", directory / "provider_state.pkl"

    def load(self, mechanical_fingerprint: str, topology_fingerprint: str):
        key = cache_key(mechanical_fingerprint, topology_fingerprint)
        manifest_path, state_path = self._paths(key)
        if not manifest_path.is_file() or not state_path.is_file():
            return None
        manifest = json.loads(manifest_path.read_text())
        payload = state_path.read_bytes()
        if manifest.get("accepted") is not True:
            raise ValueError("persistent live-topology cache contains a nonaccepted trial")
        if hashlib.sha256(payload).hexdigest() != manifest.get("state_sha256"):
            raise ValueError("live-topology cache payload hash mismatch")
        result = pickle.loads(payload)
        if result.get("topology_fingerprint") != topology_fingerprint:
            raise ValueError("live-topology cache fingerprint mismatch")
        return result

    def store_accepted(self, mechanical_fingerprint: str, result: dict[str, Any]) -> dict[str, Any]:
        topology = str(result["topology_fingerprint"])
        key = cache_key(mechanical_fingerprint, topology)
        manifest_path, state_path = self._paths(key)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = pickle.dumps(result, protocol=5)
        manifest = {
            "schema": CACHE_SCHEMA, "provider_schema": SCHEMA,
            "kernel_provider_id": PROVIDER_ID, "accepted": True,
            "cache_key": key,
            "mechanical_configuration_fingerprint": mechanical_fingerprint,
            "topology_fingerprint": topology,
            "state_sha256": hashlib.sha256(payload).hexdigest(),
            "coverage_kind": "exact_topology", "interpolation_permitted": False,
        }
        state_tmp = state_path.with_suffix(".tmp")
        manifest_tmp = manifest_path.with_suffix(".tmp")
        state_tmp.write_bytes(payload)
        manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(state_tmp, state_path)
        os.replace(manifest_tmp, manifest_path)
        return manifest

    def get_or_evaluate_accepted(
        self, mechanical_fingerprint: str, topology_fingerprint: str,
        evaluator: Callable[[], dict[str, Any]],
    ) -> tuple[dict[str, Any], bool]:
        cached = self.load(mechanical_fingerprint, topology_fingerprint)
        if cached is not None:
            return cached, True
        result = evaluator()
        if result.get("topology_fingerprint") != topology_fingerprint:
            raise ValueError("live provider returned a different exact topology")
        self.store_accepted(mechanical_fingerprint, result)
        return result, False


__all__ = ["CACHE_SCHEMA", "ExactTopologyCache", "cache_key"]
