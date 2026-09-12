"""Canonical V12 registry checkpoint and V11/V5.4.1 compatibility adapter."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
from dataclasses import dataclass
from typing import Any, Mapping

from .branch_checkpoint_v11 import ProductionBranchCheckpoint
from .directional_competition_v11 import competition_state_to_dict
from .general_multifront_v12 import (
    BranchJunctionState, FrontRuntimeState, GlobalSchedulerState,
    MultiFrontRuntimeState, ProcessEngineState, ProcessRegionState,
    ResourcePolicy, canonical_hash,
)


SCHEMA = "v12.general-multifront-checkpoint/2"
ACCEPTED_BOUNDARY_SCHEMA = "v12.accepted-boundary-checkpoint/1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checkpoint_field_hashes(
    runtime: MultiFrontRuntimeState, *, accepted_fem_state_reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose exact hashes for every production restart ownership field."""
    provenance = dict(runtime.compatibility_provenance or {})
    return {
        "accepted_fem_state_identity": canonical_hash(dict(accepted_fem_state_reference)),
        "crack_topology": canonical_hash(runtime.crack_network.to_dict()),
        "front_registry": {
            key: canonical_hash(value.to_dict()) for key, value in runtime.front_runtimes.items()
        },
        "front_competition_action_threshold_ordinal": {
            key: canonical_hash(value.competition_state)
            for key, value in runtime.front_runtimes.items()
        },
        "front_rng": {
            key: canonical_hash(value.lineage_rng_state)
            for key, value in runtime.front_runtimes.items()
        },
        "owner_registry": canonical_hash({
            "owner_by_front": runtime.owner_by_front,
            "process_regions": {
                key: value.to_dict() for key, value in runtime.process_regions.items()
            },
        }),
        "mutable_engines": {
            key: canonical_hash(value.to_dict()) for key, value in runtime.process_engines.items()
        },
        "owner_ledgers": {
            owner_id: canonical_hash({
                "active": runtime.process_engines[region.process_engine_id].active_ledgers,
                "wake": runtime.process_engines[region.process_engine_id].wake_ledgers,
                "signed": runtime.process_engines[region.process_engine_id].signed_system_ledgers,
            }) for owner_id, region in runtime.process_regions.items()
        },
        "owner_local_kernel_coordinates": canonical_hash({
            key: value.cumulative_process_advance_m
            for key, value in runtime.process_regions.items()
        }),
        "junctions": canonical_hash({
            key: value.to_dict() for key, value in runtime.junctions.items()
        }),
        "reservoirs": canonical_hash({
            key: value.to_dict() for key, value in runtime.reservoirs.items()
        }),
        "scheduler": canonical_hash(runtime.scheduler.to_dict()),
        "family_identity_and_exact_prefix_policy": canonical_hash({
            "family_identity": provenance.get("family_identity"),
            "exact_prefix_policy": provenance.get("exact_prefix_policy"),
            "restart_family_migration_provenance": provenance.get(
                "restart_family_migration_provenance"
            ),
        }),
        "output_counters": canonical_hash(runtime.output_counters or {}),
    }


def checkpoint_bytes(
    runtime: MultiFrontRuntimeState, *, accepted_fem_state_reference: Mapping[str, Any],
) -> bytes:
    payload = {
        "schema": SCHEMA,
        "accepted_fem_state_reference": json.loads(json.dumps(
            dict(accepted_fem_state_reference), sort_keys=True, allow_nan=False
        )),
        "runtime": runtime.to_dict(),
    }
    payload["runtime_sha256"] = canonical_hash(payload["runtime"])
    payload["field_hashes"] = checkpoint_field_hashes(
        runtime, accepted_fem_state_reference=payload["accepted_fem_state_reference"],
    )
    return (json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n").encode()


def write_multifront_checkpoint(
    runtime: MultiFrontRuntimeState, path: str | Path, *,
    accepted_fem_state_reference: Mapping[str, Any],
) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = checkpoint_bytes(runtime, accepted_fem_state_reference=accepted_fem_state_reference)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, target)
    return {
        "schema": SCHEMA, "path": str(target.resolve()),
        "sha256": _sha(data), "size_bytes": len(data),
        "runtime_sha256": canonical_hash(runtime.to_dict()),
    }


@dataclass(frozen=True)
class RestoredMultiFrontCheckpoint:
    runtime: MultiFrontRuntimeState
    accepted_fem_state_reference: Mapping[str, Any]
    field_hashes: Mapping[str, Any]
    accepted_fem_state: Any | None = None


@dataclass(frozen=True)
class RestoredAcceptedBoundaryCheckpoint:
    """One complete, mechanically accepted V12 restart boundary."""

    runtime: MultiFrontRuntimeState
    accepted_fem_state: Any
    accepted_stress_field: Any
    mechanics_source_identity: str
    physical_time_s: float
    accepted_opening_m: float
    step_count: int
    context_restart: Mapping[str, Any]
    payload_sha256: str


def write_accepted_boundary_checkpoint_v12(
    runtime: MultiFrontRuntimeState, path: str | Path, *,
    accepted_fem_state: Any, accepted_stress_field: Any,
    mechanics_source_identity: str, physical_time_s: float,
    accepted_opening_m: float, step_count: int,
    context_restart: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically replace one self-contained accepted-boundary checkpoint.

    The temporary file is fully written and fsynced before ``os.replace``.
    Unlike the compatibility checkpoint above, there is no FEM or context
    sidecar that can be observed from a different accepted generation.
    """
    import numpy as np

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    topology = hashlib.sha256(
        accepted_fem_state.crack_network.to_json().encode()
    ).hexdigest()
    if topology != runtime.topology_fingerprint:
        raise ValueError("accepted FEM topology differs from V12 checkpoint topology")
    sigma = np.asarray(accepted_stress_field, dtype=float)
    if sigma.ndim != 2 or not sigma.size or not np.all(np.isfinite(sigma)):
        raise ValueError("accepted-boundary checkpoint requires finite sigma_gp")
    payload = {
        "runtime": runtime.to_dict(),
        "accepted_fem_state": accepted_fem_state,
        "accepted_stress_field": sigma.copy(),
        "mechanics_source_identity": str(mechanics_source_identity),
        "physical_time_s": float(physical_time_s),
        "accepted_opening_m": float(accepted_opening_m),
        "step_count": int(step_count),
        "context_restart": dict(context_restart or {}),
    }
    payload_bytes = pickle.dumps(payload, protocol=5)
    envelope = {
        "schema": ACCEPTED_BOUNDARY_SCHEMA,
        "payload_sha256": _sha(payload_bytes),
        "payload": payload_bytes,
    }
    data = pickle.dumps(envelope, protocol=5)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    try:
        directory_fd = os.open(target.parent, os.O_RDONLY)
    except OSError:
        directory_fd = None
    if directory_fd is not None:
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return {
        "schema": ACCEPTED_BOUNDARY_SCHEMA,
        "path": str(target.resolve()), "sha256": _sha(data),
        "size_bytes": len(data), "payload_sha256": envelope["payload_sha256"],
        "runtime_sha256": canonical_hash(runtime.to_dict()),
        "accepted_state_id": runtime.accepted_state_id,
        "stress_field_state_id": runtime.stress_field_state_id,
    }


def load_accepted_boundary_checkpoint_v12(
    path: str | Path,
) -> RestoredAcceptedBoundaryCheckpoint:
    """Load and cross-check one self-contained accepted boundary."""
    import numpy as np

    envelope = pickle.loads(Path(path).read_bytes())
    if envelope.get("schema") != ACCEPTED_BOUNDARY_SCHEMA:
        raise ValueError("unsupported V12 accepted-boundary checkpoint schema")
    payload_bytes = envelope.get("payload")
    if not isinstance(payload_bytes, bytes) or _sha(payload_bytes) != envelope.get(
        "payload_sha256"
    ):
        raise ValueError("V12 accepted-boundary payload hash mismatch")
    payload = pickle.loads(payload_bytes)
    runtime = MultiFrontRuntimeState.from_dict(payload["runtime"])
    accepted = payload["accepted_fem_state"]
    topology = hashlib.sha256(accepted.crack_network.to_json().encode()).hexdigest()
    if topology != runtime.topology_fingerprint:
        raise ValueError("accepted FEM topology differs from V12 checkpoint topology")
    sigma = np.asarray(payload["accepted_stress_field"], dtype=float)
    if sigma.ndim != 2 or not sigma.size or not np.all(np.isfinite(sigma)):
        raise ValueError("accepted-boundary checkpoint contains invalid sigma_gp")
    from .stateful_multifront_production_v12 import (
        accepted_context_state_identity, stress_field_identity,
    )
    accepted_id = accepted_context_state_identity(
        accepted, runtime, physical_time_s=payload["physical_time_s"],
        accepted_opening_m=payload["accepted_opening_m"],
        step_count=payload["step_count"],
    )
    if accepted_id != runtime.accepted_state_id:
        raise ValueError("accepted-boundary FEM/time/load identity mismatch")
    stress_id = stress_field_identity(
        accepted, sigma,
        mechanics_source_identity=payload["mechanics_source_identity"],
        accepted_state_id=accepted_id,
    )
    if stress_id != runtime.stress_field_state_id:
        raise ValueError("accepted-boundary stress identity mismatch")
    sigma = sigma.copy(); sigma.setflags(write=False)
    return RestoredAcceptedBoundaryCheckpoint(
        runtime=runtime, accepted_fem_state=accepted,
        accepted_stress_field=sigma,
        mechanics_source_identity=payload["mechanics_source_identity"],
        physical_time_s=float(payload["physical_time_s"]),
        accepted_opening_m=float(payload["accepted_opening_m"]),
        step_count=int(payload["step_count"]),
        context_restart=dict(payload.get("context_restart", {})),
        payload_sha256=envelope["payload_sha256"],
    )


def load_multifront_checkpoint(path: str | Path) -> RestoredMultiFrontCheckpoint:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported V12 multi-front checkpoint schema")
    if canonical_hash(payload.get("runtime")) != payload.get("runtime_sha256"):
        raise ValueError("V12 runtime checkpoint hash mismatch")
    runtime = MultiFrontRuntimeState.from_dict(payload["runtime"])
    expected = checkpoint_field_hashes(
        runtime, accepted_fem_state_reference=payload["accepted_fem_state_reference"],
    )
    if payload.get("field_hashes") != expected:
        raise ValueError("V12 checkpoint per-field hash mismatch")
    return RestoredMultiFrontCheckpoint(
        runtime=runtime,
        accepted_fem_state_reference=payload["accepted_fem_state_reference"],
        field_hashes=expected,
    )


def restore_multifront_checkpoint(path: str | Path) -> MultiFrontRuntimeState:
    return load_multifront_checkpoint(path).runtime


def write_production_multifront_checkpoint(
    runtime: MultiFrontRuntimeState, path: str | Path, *, accepted_fem_state: Any,
) -> dict[str, Any]:
    """Atomically persist V12 registries plus the exact accepted FEM object.

    The accepted FEM object remains a binary sidecar because its numpy arrays
    are not JSON data.  The JSON manifest owns the immutable sidecar identity.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fem_data = pickle.dumps(accepted_fem_state, protocol=5)
    fem_name = target.name + ".accepted-fem.pkl"
    reference = {
        "kind": "immutable_accepted_fem_state_backing",
        "file": fem_name,
        "sha256": _sha(fem_data),
        "size_bytes": len(fem_data),
        "topology_fingerprint": runtime.topology_fingerprint,
        "accepted_state_id": runtime.accepted_state_id,
        "stress_field_state_id": runtime.stress_field_state_id,
    }
    data = checkpoint_bytes(runtime, accepted_fem_state_reference=reference)
    fem_target = target.with_name(fem_name)
    fem_tmp = fem_target.with_name(fem_target.name + ".tmp")
    manifest_tmp = target.with_name(target.name + ".tmp")
    fem_tmp.write_bytes(fem_data)
    manifest_tmp.write_bytes(data)
    os.replace(fem_tmp, fem_target)
    os.replace(manifest_tmp, target)
    return {
        "schema": SCHEMA, "path": str(target.resolve()),
        "sha256": _sha(data), "size_bytes": len(data),
        "accepted_fem_state_sha256": reference["sha256"],
        "accepted_fem_state_size_bytes": len(fem_data),
        "runtime_sha256": canonical_hash(runtime.to_dict()),
    }


def load_production_multifront_checkpoint(
    path: str | Path,
) -> RestoredMultiFrontCheckpoint:
    target = Path(path)
    restored = load_multifront_checkpoint(target)
    reference = restored.accepted_fem_state_reference
    if reference.get("kind") != "immutable_accepted_fem_state_backing":
        raise ValueError("checkpoint does not own a production FEM backing object")
    fem_path = target.with_name(str(reference["file"]))
    data = fem_path.read_bytes()
    if _sha(data) != reference.get("sha256") or len(data) != reference.get("size_bytes"):
        raise ValueError("accepted FEM backing object hash or size mismatch")
    accepted = pickle.loads(data)
    physical_topology = hashlib.sha256(
        accepted.crack_network.to_json().encode()
    ).hexdigest()
    if physical_topology != restored.runtime.topology_fingerprint:
        raise ValueError("accepted FEM topology differs from V12 checkpoint topology")
    return RestoredMultiFrontCheckpoint(
        runtime=restored.runtime,
        accepted_fem_state_reference=reference,
        field_hashes=restored.field_hashes,
        accepted_fem_state=accepted,
    )


def _opaque_fingerprint(value: Any) -> str:
    return _sha(pickle.dumps(value, protocol=5))


def _process_coordinate(shared: Mapping[str, Any]) -> float:
    mpz = shared.get("mpz_fields", {})
    if isinstance(mpz, Mapping):
        return float(mpz.get("advance_total_m", 0.0))
    return 0.0


def _engine_state_from_payload(
    *, engine_id: str, source_id: str, payload: Mapping[str, Any],
    active_ledgers: Mapping[str, float], update_count: int,
    renewal_count: int, coordinate: float, family_identity: str,
) -> ProcessEngineState:
    if payload.get("schema") == "v11.shared-production-engine-state/1":
        return ProcessEngineState.from_v11_payload(
            engine_id=engine_id, source_state_id=source_id, payload=payload,
            active_ledgers=active_ledgers, wake_ledgers={}, signed_system_ledgers={},
            update_count=update_count, event_renewal_count=renewal_count,
            local_process_coordinate_m=coordinate, family_identity=family_identity,
        )
    # Retain compatibility with source-only legacy unit fixtures.  Production
    # qualification explicitly rejects these opaque-only objects.
    return ProcessEngineState(
        engine_id=engine_id, source_state_id=source_id,
        active_ledgers=active_ledgers, wake_ledgers={}, signed_system_ledgers={},
        update_count=update_count, event_renewal_count=renewal_count,
        local_process_coordinate_m=coordinate,
        opaque_state_fingerprint=_opaque_fingerprint(payload),
        mutable_state={"compatibility_only_incomplete_engine_payload": True},
    )


def load_v11_checkpoint_as_v12(
    checkpoint: ProductionBranchCheckpoint, *,
    resource_policy: ResourcePolicy,
) -> MultiFrontRuntimeState:
    """Map a frozen V11 one-owner state without changing physical/stochastic data.

    The accepted FEM object remains owned by the caller.  V12 stores its exact
    identity and the complete V11 competition dictionaries; it does not solve,
    migrate, or mutate the accepted state.
    """
    network = checkpoint.state.crack_network
    active = set(network.active_tip_ids)
    front_runtimes = {}
    for front_id, competition in checkpoint.front_competitions.items():
        inventory = tuple(item.candidate_id for item in competition.candidates)
        assigned = network.branch(front_id).local_state.get("candidate_id")
        mechanically_active = (
            (str(assigned),) if assigned in inventory else inventory
        )
        front_runtimes[front_id] = FrontRuntimeState(
            front_id=front_id,
            competition_state=competition_state_to_dict(competition),
            candidate_ids=inventory,
            lineage_rng_state={
                "global_hazard_seed": competition.global_hazard_seed,
                "competition_event_index": competition.competition_event_index,
                "v11_rng_state_fingerprint": _opaque_fingerprint(checkpoint.state.rng_state),
            },
            interval_count=int(checkpoint.state.event_counters.get("accepted_steps", 0)),
            mechanically_active_candidate_ids=mechanically_active,
        )

    junctions = {}
    owner_by_front: dict[str, str] = {}
    regions: dict[str, ProcessRegionState] = {}
    engines: dict[str, ProcessEngineState] = {}
    claimed: set[str] = set()
    shared_fingerprint = _opaque_fingerprint(checkpoint.shared_process_state)
    coordinate = _process_coordinate(checkpoint.shared_process_state)

    bundle = checkpoint.shared_process_state
    bundle_owner_by_front = None
    if bundle.get("schema") == "v11.multi-tip-engine-bundle/2":
        ownership = bundle.get("ownership_registry", {})
        bundle_owner_by_front = {
            str(front): str(owner)
            for front, owner in ownership.get("owner_by_tip", {}).items()
        }
        if set(bundle_owner_by_front) != active:
            raise ValueError("V11 multi-engine bundle omits an active front owner")
        cluster_by_id = {
            item.cluster_id: item for item in checkpoint.branch_clusters
        }
        for cluster in checkpoint.branch_clusters:
            junctions[cluster.cluster_id] = BranchJunctionState(
                junction_id=cluster.cluster_id,
                parent_branch_id=cluster.parent_branch_id,
                child_branch_ids=cluster.arm_branch_ids,
                birth_transaction_id=(
                    f"v11:event:{network.branch(cluster.arm_branch_ids[0]).initiation_event}"
                ),
                junction_xy_m=cluster.junction_xy_m,
                status="unresolved" if cluster.unresolved else "resolved",
            )
        for source_owner in sorted(set(bundle_owner_by_front.values())):
            members = frozenset(
                front for front, owner in bundle_owner_by_front.items()
                if owner == source_owner
            )
            payload = bundle.get("engines", {}).get(source_owner)
            if not isinstance(payload, Mapping):
                raise ValueError("V11 multi-engine bundle omits a physical engine")
            engine_id = f"engine:v11:{source_owner}"
            owner_id = f"owner:v11:{source_owner}"
            source_id = canonical_hash({
                "bundle_engine_owner": source_owner,
                "engine_payload_sha256": _opaque_fingerprint(payload),
            })
            local_coordinate = _process_coordinate(payload)
            cluster = cluster_by_id.get(source_owner)
            ledgers = {} if cluster is None else cluster.conserved_ledgers
            engine = _engine_state_from_payload(
                engine_id=engine_id, source_id=source_id, payload=payload,
                active_ledgers=ledgers,
                update_count=int(checkpoint.state.event_counters.get(
                    "shared_state_updates", 0
                )),
                renewal_count=int(checkpoint.state.event_counters.get(
                    "topology_actions", 0
                )),
                coordinate=local_coordinate,
                family_identity=checkpoint.provider_cache_identity,
            )
            unresolved = frozenset(
                item.cluster_id for item in checkpoint.branch_clusters
                if item.unresolved and set(item.arm_branch_ids).intersection(members)
            )
            engines[engine_id] = engine
            regions[owner_id] = ProcessRegionState(
                owner_id=owner_id, member_front_ids=members,
                unresolved_junction_ids=unresolved,
                cumulative_process_advance_m=local_coordinate,
                process_engine_id=engine_id, source_state_id=source_id,
            )
            for front_id in members:
                owner_by_front[front_id] = owner_id
                if not unresolved:
                    front_runtimes[front_id] = front_runtimes[
                        front_id
                    ].activate_complete_competition()
        claimed.update(active)

    for cluster in sorted(
        (() if bundle_owner_by_front is not None else checkpoint.branch_clusters),
        key=lambda item: item.cluster_id,
    ):
        member_fronts = frozenset(active.intersection(cluster.arm_branch_ids))
        if not member_fronts:
            continue
        if claimed.intersection(member_fronts):
            raise ValueError("V11 compatibility state assigns one front to multiple clusters")
        claimed.update(member_fronts)
        junctions[cluster.cluster_id] = BranchJunctionState(
            junction_id=cluster.cluster_id,
            parent_branch_id=cluster.parent_branch_id,
            child_branch_ids=cluster.arm_branch_ids,
            birth_transaction_id=f"v11:event:{network.branch(cluster.arm_branch_ids[0]).initiation_event}",
            junction_xy_m=cluster.junction_xy_m,
            status="unresolved" if cluster.unresolved else "retired",
        )
        owner_id = f"owner:v11:{cluster.cluster_id}"
        engine_id = f"engine:v11:{cluster.cluster_id}"
        source_id = canonical_hash({
            "shared_process_state_fingerprint": _opaque_fingerprint(cluster.shared_process_state),
            "cluster_id": cluster.cluster_id,
        })
        engine = _engine_state_from_payload(
            engine_id=engine_id, source_id=source_id,
            # ``BranchClusterState.shared_process_state`` is a compact audit
            # summary.  The complete physical engine+MPZ owner is the
            # checkpoint-level shared_process_state captured by V11.
            payload=checkpoint.shared_process_state,
            active_ledgers=cluster.conserved_ledgers,
            update_count=int(checkpoint.state.event_counters.get("shared_state_updates", 0)),
            renewal_count=int(checkpoint.state.event_counters.get("topology_actions", 0)),
            coordinate=coordinate,
            family_identity=checkpoint.provider_cache_identity,
        )
        engines[engine_id] = engine
        regions[owner_id] = ProcessRegionState(
            owner_id=owner_id, member_front_ids=member_fronts,
            unresolved_junction_ids=(frozenset({cluster.cluster_id}) if cluster.unresolved else frozenset()),
            cumulative_process_advance_m=coordinate, process_engine_id=engine_id,
            source_state_id=source_id,
        )
        owner_by_front.update({front_id: owner_id for front_id in member_fronts})

    unclaimed = sorted(active - claimed)
    if unclaimed:
        if len(unclaimed) > 1 and checkpoint.branch_clusters:
            raise ValueError("V11 compatibility cannot infer independent engines absent owner records")
        for front_id in unclaimed:
            owner_id = f"owner:v11:{front_id}"
            engine_id = f"engine:v11:{front_id}"
            source_id = canonical_hash({
                "shared_process_state_fingerprint": shared_fingerprint,
                "front_id": front_id,
            })
            engine = _engine_state_from_payload(
                engine_id=engine_id, source_id=source_id,
                payload=checkpoint.shared_process_state,
                active_ledgers={},
                update_count=int(checkpoint.state.event_counters.get("shared_state_updates", 0)),
                renewal_count=int(checkpoint.state.event_counters.get("topology_actions", 0)),
                coordinate=coordinate,
                family_identity=checkpoint.provider_cache_identity,
            )
            engines[engine_id] = engine
            regions[owner_id] = ProcessRegionState(
                owner_id=owner_id, member_front_ids=frozenset({front_id}),
                unresolved_junction_ids=frozenset(), cumulative_process_advance_m=coordinate,
                process_engine_id=engine_id, source_state_id=source_id,
            )
            owner_by_front[front_id] = owner_id

    physical_identity = _opaque_fingerprint(checkpoint.state)
    accepted_state_id = checkpoint.topology_fingerprint
    stress_state_id = str(
        checkpoint.state.junction_process_state.get("accepted_stress_state_id", accepted_state_id)
    )
    runtime = MultiFrontRuntimeState(
        crack_network=network, front_runtimes=front_runtimes,
        owner_by_front=owner_by_front, process_regions=regions,
        process_engines=engines, junctions=junctions, reservoirs={},
        scheduler=GlobalSchedulerState(
            accepted_interval_index=int(checkpoint.state.event_counters.get("accepted_steps", 0)),
            accepted_transaction_index=int(checkpoint.state.event_counters.get("topology_actions", 0)),
            global_hazard_seed=min(
                item.global_hazard_seed for item in checkpoint.front_competitions.values()
            ),
            competition_event_index=sum(
                item.competition_event_index for item in checkpoint.front_competitions.values()
            ),
        ),
        resource_policy=resource_policy,
        accepted_state_id=accepted_state_id, stress_field_state_id=stress_state_id,
        compatibility_provenance={
            "source_schema": "v11.production-branch-network-checkpoint/2",
            "source_topology_fingerprint": checkpoint.topology_fingerprint,
            "source_provider_cache_identity": checkpoint.provider_cache_identity,
            "family_identity": checkpoint.provider_cache_identity,
            "exact_prefix_policy": "v5_4_2_exact_prefix_immutable",
            "restart_family_migration_provenance": (
                None if checkpoint.restart_family_migration_provenance is None
                else dict(checkpoint.restart_family_migration_provenance)
            ),
        },
        cumulative_branch_births=int(checkpoint.state.event_counters.get(
            "committed_branch_birth_count", len(junctions)
        )),
        cumulative_coalescences=int(checkpoint.state.event_counters.get("coalescence_count", 0)),
        cumulative_retirements=int(checkpoint.state.event_counters.get("retirement_count", 0)),
        output_counters={
            "accepted_steps": int(checkpoint.state.event_counters.get("accepted_steps", 0)),
            "topology_actions": int(checkpoint.state.event_counters.get("topology_actions", 0)),
        },
    )
    # The fingerprint is intentionally computed after construction.  Merely
    # reading a V11 checkpoint must never mutate its accepted FEM state.
    if _opaque_fingerprint(checkpoint.state) != physical_identity:
        raise RuntimeError("V11 compatibility loading mutated accepted physical state")
    return runtime


__all__ = [
    "SCHEMA", "RestoredMultiFrontCheckpoint", "checkpoint_bytes", "checkpoint_field_hashes",
    "load_multifront_checkpoint", "load_v11_checkpoint_as_v12",
    "restore_multifront_checkpoint", "write_multifront_checkpoint",
    "load_production_multifront_checkpoint", "write_production_multifront_checkpoint",
]
