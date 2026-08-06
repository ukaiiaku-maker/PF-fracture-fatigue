"""Atomic production restart contract for bounded v11 branch networks."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
from typing import Any, Mapping

from .branch_cluster_v11 import BranchClusterState
from .directional_competition_v11 import (
    DirectionalCompetitionState, competition_state_to_dict,
)
from .topology_transaction_v11 import LiveFEMTopologyState


SCHEMA = "v11.production-branch-network-checkpoint/2"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class ProductionBranchCheckpoint:
    state: LiveFEMTopologyState
    shared_process_state: Mapping[str, Any]
    physical_time_s: float
    accepted_load: float
    mesh_identity: str
    boundary_condition_state: Mapping[str, Any]
    provider_runtime: Any
    provider_cache_identity: str
    topology_fingerprint: str
    front_competitions: Mapping[str, DirectionalCompetitionState]
    branch_clusters: tuple[BranchClusterState, ...]
    projected_extension_m: float
    physical_extension_m: float
    handoff_guard_diagnostics: Mapping[str, Any]
    termination_reason: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "physical_time_s", "accepted_load", "projected_extension_m",
            "physical_extension_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        if not self.mesh_identity or not self.provider_cache_identity or not self.topology_fingerprint:
            raise ValueError("mesh, provider-cache, and topology identities are required")
        active = set(self.state.crack_network.active_tip_ids)
        competitions = set(self.front_competitions)
        if competitions != active:
            raise ValueError("front competitions must map one-to-one to active fronts")
        if len({item.cluster_id for item in self.branch_clusters}) != len(self.branch_clusters):
            raise ValueError("duplicate branch cluster checkpoint entry")

    def manifest_fields(self) -> dict[str, Any]:
        runtime = self.provider_runtime
        return {
            "physical_time_s": self.physical_time_s,
            "accepted_load": self.accepted_load,
            "mesh_identity": self.mesh_identity,
            "boundary_condition_state": dict(self.boundary_condition_state),
            "mechanics_provider": (
                runtime.routing.active_mechanics_provider if runtime is not None else None
            ),
            "provider_transition_state": (
                runtime.audit_payload() if runtime is not None else None
            ),
            "provider_cache_identity": self.provider_cache_identity,
            "topology_fingerprint": self.topology_fingerprint,
            "crack_network": self.state.crack_network.to_dict(),
            "active_front_ids": list(self.state.crack_network.active_tip_ids),
            "front_competitions": {
                key: competition_state_to_dict(value)
                for key, value in sorted(self.front_competitions.items())
            },
            "branch_cluster_ids": [item.cluster_id for item in self.branch_clusters],
            "projected_extension_m": self.projected_extension_m,
            "physical_extension_m": self.physical_extension_m,
            "event_counters": dict(self.state.event_counters),
            "energy_ledgers": dict(self.state.energy_ledgers),
            "handoff_guard_diagnostics": dict(self.handoff_guard_diagnostics),
            "termination_reason": self.termination_reason,
            "has_rng_state": self.state.rng_state is not None,
            "shared_process_engine_type": str(self.shared_process_state.get("engine_type", "unknown")),
            "fem_reconstruction": {
                "displacement_shape": list(self.state.displacement.shape),
                "damage_shape": list(self.state.damage.shape),
                "ep_gp_shape": list(self.state.ep_gp.shape),
                "rho_gp_shape": list(self.state.rho_gp.shape),
            },
        }


def write_branch_checkpoint(
    checkpoint: ProductionBranchCheckpoint, path: str | Path,
) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = pickle.dumps(checkpoint, protocol=5)
    state_name = target.name + ".state.pkl"
    manifest = {
        "schema": SCHEMA, "state_file": state_name, "state_sha256": _sha(data),
        **checkpoint.manifest_fields(),
    }
    state_target = target.with_name(state_name)
    state_tmp = state_target.with_name(state_target.name + ".tmp")
    manifest_tmp = target.with_name(target.name + ".tmp")
    state_tmp.write_bytes(data)
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(state_tmp, state_target)
    os.replace(manifest_tmp, target)
    return manifest


def restore_branch_checkpoint(path: str | Path) -> ProductionBranchCheckpoint:
    target = Path(path)
    manifest = json.loads(target.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported production branch checkpoint schema")
    data = target.with_name(manifest["state_file"]).read_bytes()
    if _sha(data) != manifest.get("state_sha256"):
        raise ValueError("production branch checkpoint state hash mismatch")
    checkpoint = pickle.loads(data)
    if not isinstance(checkpoint, ProductionBranchCheckpoint):
        raise ValueError("production branch checkpoint payload has the wrong type")
    if checkpoint.manifest_fields() != {
        key: value for key, value in manifest.items()
        if key not in {"schema", "state_file", "state_sha256"}
    }:
        raise ValueError("production branch checkpoint manifest does not match state")
    return checkpoint


__all__ = [
    "ProductionBranchCheckpoint", "SCHEMA", "restore_branch_checkpoint",
    "write_branch_checkpoint",
]
