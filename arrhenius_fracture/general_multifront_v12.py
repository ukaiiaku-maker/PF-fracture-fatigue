"""Front-count-independent V12 topology and process-region runtime.

This module is deliberately source-only.  It owns no constitutive parameters,
does not call the FEM provider, and never advances a stochastic clock by
itself.  It defines the registries and atomic transactions consumed by a future
production driver.  Fracture transactions remain binary: one existing front
advances, or one existing front is replaced by two daughters.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import base64
import copy
import hashlib
import json
import math
import pickle
from typing import Any, Callable, Iterable, Mapping, Sequence

from .crack_network_v11 import CrackBranchState, CrackNetworkState


SCHEMA = "v12.general-multifront-runtime/1"
POLICY_BOUND_FRONT_LIMIT = "configured_front_resource_limit_reached"
POLICY_BOUND_TRANSACTION_LIMIT = "configured_branch_transaction_limit_reached"
PARENT_PROCESS_ZONE_UNRESOLVED = "parent_process_zone_still_unresolved"
JUNCTION_STATUSES = frozenset({"unresolved", "resolved", "retired"})
BRANCHING_MODES = frozenset({"disabled", "mechanistic"})
SHARED_REGION_BRANCHING_MODES = frozenset({"forbid", "experimental"})
SCHEDULER_POLICIES = frozenset({
    "v11_correlated_proposal_compatibility",
    "v12_global_earliest_proposal",
})
TRANSACTION_KINDS = frozenset({
    "one_arm", "two_arm", "coalescence", "retirement",
    "junction_resolution", "process_region_partition",
})


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return json.loads(canonical_json(dict(value or {})))


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _finite_point(value: Iterable[float], name: str) -> tuple[float, float]:
    point = tuple(float(item) for item in value)
    if len(point) != 2 or not all(math.isfinite(item) for item in point):
        raise ValueError(f"{name} must contain two finite coordinates")
    return point  # type: ignore[return-value]


def _sorted_ids(values: Iterable[str], name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    result = tuple(sorted(str(item) for item in values))
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicate identifiers")
    return result


def _ledger(value: Mapping[str, float] | None) -> dict[str, float]:
    result = {str(key): _finite_nonnegative(item, f"ledger[{key}]") for key, item in (value or {}).items()}
    return dict(sorted(result.items()))


def _signed_ledger(value: Mapping[str, float] | None) -> dict[str, float]:
    result = {}
    for key, item in (value or {}).items():
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"signed_ledger[{key}] must be finite")
        result[str(key)] = number
    return dict(sorted(result.items()))


def _ledger_sum(*values: Mapping[str, float]) -> dict[str, float]:
    keys = sorted(set().union(*(item.keys() for item in values)))
    return {key: math.fsum(float(item.get(key, 0.0)) for item in values) for key in keys}


@dataclass(frozen=True)
class BranchJunctionState:
    junction_id: str
    parent_branch_id: str
    child_branch_ids: tuple[str, str]
    birth_transaction_id: str
    junction_xy_m: tuple[float, float]
    status: str = "unresolved"
    reservoir_id: str | None = None

    def __post_init__(self) -> None:
        children = _sorted_ids(self.child_branch_ids, "child_branch_ids")
        if len(children) != 2:
            raise ValueError("a physical junction must have exactly two children")
        if self.parent_branch_id in children:
            raise ValueError("a junction parent cannot be its own child")
        if self.status not in JUNCTION_STATUSES:
            raise ValueError(f"unsupported junction status: {self.status}")
        object.__setattr__(self, "child_branch_ids", children)
        object.__setattr__(self, "junction_xy_m", _finite_point(self.junction_xy_m, "junction_xy_m"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "junction_id": self.junction_id,
            "parent_branch_id": self.parent_branch_id,
            "child_branch_ids": list(self.child_branch_ids),
            "birth_transaction_id": self.birth_transaction_id,
            "junction_xy_m": list(self.junction_xy_m),
            "status": self.status,
            "reservoir_id": self.reservoir_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BranchJunctionState":
        return cls(
            junction_id=value["junction_id"], parent_branch_id=value["parent_branch_id"],
            child_branch_ids=tuple(value["child_branch_ids"]),
            birth_transaction_id=value["birth_transaction_id"],
            junction_xy_m=tuple(value["junction_xy_m"]), status=value["status"],
            reservoir_id=value.get("reservoir_id"),
        )


@dataclass(frozen=True)
class ProcessEngineState:
    engine_id: str
    source_state_id: str
    active_ledgers: Mapping[str, float]
    wake_ledgers: Mapping[str, float]
    signed_system_ledgers: Mapping[str, float]
    update_count: int = 0
    event_renewal_count: int = 0
    local_process_coordinate_m: float = 0.0
    opaque_state_fingerprint: str = "fresh"
    mutable_state: Mapping[str, Any] | None = None
    rng_state: Mapping[str, Any] | None = None
    checkpoint_payload_b64: str | None = None
    checkpoint_payload_sha256: str | None = None
    engine_class: str | None = None
    engine_model_id: str | None = None
    checkpoint_field_inventory: tuple[str, ...] = ()
    family_identity: str | None = None

    def __post_init__(self) -> None:
        if not self.engine_id or not self.source_state_id:
            raise ValueError("process engine and source-state identities are required")
        if int(self.update_count) < 0 or int(self.event_renewal_count) < 0:
            raise ValueError("process update and renewal counts must be nonnegative")
        object.__setattr__(self, "update_count", int(self.update_count))
        object.__setattr__(self, "event_renewal_count", int(self.event_renewal_count))
        object.__setattr__(self, "local_process_coordinate_m", _finite_nonnegative(
            self.local_process_coordinate_m, "local_process_coordinate_m"
        ))
        object.__setattr__(self, "active_ledgers", _ledger(self.active_ledgers))
        object.__setattr__(self, "wake_ledgers", _ledger(self.wake_ledgers))
        object.__setattr__(
            self, "signed_system_ledgers",
            _signed_ledger(self.signed_system_ledgers),
        )
        object.__setattr__(self, "mutable_state", _mapping(self.mutable_state))
        object.__setattr__(self, "rng_state", _mapping(self.rng_state))
        inventory = tuple(sorted(str(item) for item in self.checkpoint_field_inventory))
        if len(inventory) != len(set(inventory)):
            raise ValueError("process-engine checkpoint field inventory contains duplicates")
        object.__setattr__(self, "checkpoint_field_inventory", inventory)
        if self.checkpoint_payload_b64 is not None:
            try:
                payload = base64.b64decode(self.checkpoint_payload_b64, validate=True)
            except Exception as exc:
                raise ValueError("invalid process-engine checkpoint payload") from exc
            digest = hashlib.sha256(payload).hexdigest()
            if self.checkpoint_payload_sha256 != digest:
                raise ValueError("process-engine checkpoint payload hash mismatch")
            restored = pickle.loads(payload)
            if not isinstance(restored, Mapping):
                raise ValueError("process-engine checkpoint payload must restore a mapping")
            if restored.get("schema") != "v11.shared-production-engine-state/1":
                raise ValueError("unsupported complete process-engine checkpoint schema")
            if self.engine_class != restored.get("engine_type"):
                raise ValueError("process-engine class differs from complete checkpoint payload")
        elif any((self.checkpoint_payload_sha256, self.engine_class, inventory)):
            raise ValueError("complete process-engine metadata requires checkpoint payload bytes")

    @classmethod
    def from_v11_payload(
        cls, *, engine_id: str, source_state_id: str, payload: Mapping[str, Any],
        active_ledgers: Mapping[str, float], wake_ledgers: Mapping[str, float],
        signed_system_ledgers: Mapping[str, float], update_count: int,
        event_renewal_count: int, local_process_coordinate_m: float,
        family_identity: str | None = None,
    ) -> "ProcessEngineState":
        """Own the complete, immutable V11 engine+MPZ restart payload."""
        if payload.get("schema") != "v11.shared-production-engine-state/1":
            raise ValueError("unsupported V11 process-engine checkpoint payload")
        frozen = pickle.dumps(copy.deepcopy(dict(payload)), protocol=5)
        engine_fields = tuple(f"engine.{key}" for key in payload.get("engine_fields", {}))
        mpz_fields = tuple(f"mpz.{key}" for key in payload.get("mpz_fields", {}))
        rng_fields = {
            key: value for key, value in payload.get("engine_fields", {}).items()
            if "rng" in key.lower() or "threshold" in key.lower()
        }
        return cls(
            engine_id=engine_id, source_state_id=source_state_id,
            active_ledgers=active_ledgers, wake_ledgers=wake_ledgers,
            signed_system_ledgers=signed_system_ledgers,
            update_count=update_count, event_renewal_count=event_renewal_count,
            local_process_coordinate_m=local_process_coordinate_m,
            opaque_state_fingerprint=hashlib.sha256(frozen).hexdigest(),
            mutable_state={
                "engine_field_count": len(engine_fields),
                "mpz_field_count": len(mpz_fields),
                "mpz_type": payload.get("mpz_type"),
            },
            rng_state={
                "checkpoint_rng_sha256": hashlib.sha256(
                    pickle.dumps(rng_fields, protocol=5)
                ).hexdigest()
            },
            checkpoint_payload_b64=base64.b64encode(frozen).decode("ascii"),
            checkpoint_payload_sha256=hashlib.sha256(frozen).hexdigest(),
            engine_class=str(payload.get("engine_type")),
            engine_model_id=str(payload.get("engine_fields", {}).get(
                "MODEL_ID", payload.get("engine_type")
            )),
            checkpoint_field_inventory=engine_fields + mpz_fields,
            family_identity=family_identity,
        )

    def complete_checkpoint_payload(self) -> dict[str, Any]:
        if self.checkpoint_payload_b64 is None:
            raise RuntimeError("process owner has no complete engine checkpoint payload")
        raw = base64.b64decode(self.checkpoint_payload_b64, validate=True)
        if hashlib.sha256(raw).hexdigest() != self.checkpoint_payload_sha256:
            raise RuntimeError("process-engine checkpoint payload failed hash verification")
        return copy.deepcopy(dict(pickle.loads(raw)))

    def restore_into(self, engine: Any) -> Any:
        """Restore the exact physical state into a fresh engine of the same class."""
        from .sharp_front_v11_branching import _restore_shared_engine
        if type(engine).__name__ != self.engine_class:
            raise RuntimeError("fresh engine class differs from checkpoint owner")
        return _restore_shared_engine(engine, self.complete_checkpoint_payload())

    @property
    def conserved_ledgers(self) -> dict[str, float]:
        return _ledger_sum(self.active_ledgers, self.wake_ledgers)

    def evolve_once(self) -> "ProcessEngineState":
        return replace(self, update_count=self.update_count + 1)

    def renew(self, distance_m: float) -> "ProcessEngineState":
        distance = _finite_nonnegative(distance_m, "realized renewal distance")
        if distance <= 0.0:
            raise ValueError("an event renewal must have positive realized length")
        return replace(
            self,
            event_renewal_count=self.event_renewal_count + 1,
            local_process_coordinate_m=self.local_process_coordinate_m + distance,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_id": self.engine_id, "source_state_id": self.source_state_id,
            "active_ledgers": dict(self.active_ledgers),
            "wake_ledgers": dict(self.wake_ledgers),
            "signed_system_ledgers": dict(self.signed_system_ledgers),
            "update_count": self.update_count,
            "event_renewal_count": self.event_renewal_count,
            "local_process_coordinate_m": self.local_process_coordinate_m,
            "opaque_state_fingerprint": self.opaque_state_fingerprint,
            "mutable_state": dict(self.mutable_state or {}),
            "rng_state": dict(self.rng_state or {}),
            "checkpoint_payload_b64": self.checkpoint_payload_b64,
            "checkpoint_payload_sha256": self.checkpoint_payload_sha256,
            "engine_class": self.engine_class,
            "engine_model_id": self.engine_model_id,
            "checkpoint_field_inventory": list(self.checkpoint_field_inventory),
            "family_identity": self.family_identity,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProcessEngineState":
        return cls(**value)


@dataclass(frozen=True)
class ProcessRegionState:
    owner_id: str
    member_front_ids: frozenset[str]
    unresolved_junction_ids: frozenset[str]
    cumulative_process_advance_m: float
    process_engine_id: str
    source_state_id: str
    generation: int = 0
    source_reservoir_id: str | None = None
    detached_from_owner_id: str | None = None

    def __post_init__(self) -> None:
        members = frozenset(str(item) for item in self.member_front_ids)
        if not members:
            raise ValueError("a process region must own at least one active front")
        if not self.owner_id or not self.process_engine_id or not self.source_state_id:
            raise ValueError("process-region identities must not be empty")
        if int(self.generation) < 0:
            raise ValueError("process-region generation must be nonnegative")
        object.__setattr__(self, "member_front_ids", members)
        object.__setattr__(self, "unresolved_junction_ids", frozenset(
            str(item) for item in self.unresolved_junction_ids
        ))
        object.__setattr__(self, "generation", int(self.generation))
        object.__setattr__(self, "cumulative_process_advance_m", _finite_nonnegative(
            self.cumulative_process_advance_m, "cumulative_process_advance_m"
        ))

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner_id": self.owner_id,
            "member_front_ids": sorted(self.member_front_ids),
            "unresolved_junction_ids": sorted(self.unresolved_junction_ids),
            "cumulative_process_advance_m": self.cumulative_process_advance_m,
            "process_engine_id": self.process_engine_id,
            "source_state_id": self.source_state_id,
            "generation": self.generation,
            "source_reservoir_id": self.source_reservoir_id,
            "detached_from_owner_id": self.detached_from_owner_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProcessRegionState":
        return cls(
            owner_id=value["owner_id"], member_front_ids=frozenset(value["member_front_ids"]),
            unresolved_junction_ids=frozenset(value["unresolved_junction_ids"]),
            cumulative_process_advance_m=value["cumulative_process_advance_m"],
            process_engine_id=value["process_engine_id"], source_state_id=value["source_state_id"],
            generation=value.get("generation", 0),
            source_reservoir_id=value.get("source_reservoir_id"),
            detached_from_owner_id=value.get("detached_from_owner_id"),
        )


@dataclass(frozen=True)
class ProcessRegionReservoir:
    reservoir_id: str
    archived_owner_id: str
    archived_engine: ProcessEngineState
    member_front_ids_at_archive: tuple[str, ...]
    junction_ids: tuple[str, ...]
    archive_transaction_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "member_front_ids_at_archive", _sorted_ids(
            self.member_front_ids_at_archive, "member_front_ids_at_archive"
        ))
        object.__setattr__(self, "junction_ids", _sorted_ids(
            self.junction_ids, "junction_ids", allow_empty=True
        ))
        if not self.reservoir_id or not self.archived_owner_id or not self.archive_transaction_id:
            raise ValueError("reservoir identities must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "reservoir_id": self.reservoir_id,
            "archived_owner_id": self.archived_owner_id,
            "archived_engine": self.archived_engine.to_dict(),
            "member_front_ids_at_archive": list(self.member_front_ids_at_archive),
            "junction_ids": list(self.junction_ids),
            "archive_transaction_id": self.archive_transaction_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProcessRegionReservoir":
        return cls(
            reservoir_id=value["reservoir_id"], archived_owner_id=value["archived_owner_id"],
            archived_engine=ProcessEngineState.from_dict(value["archived_engine"]),
            member_front_ids_at_archive=tuple(value["member_front_ids_at_archive"]),
            junction_ids=tuple(value["junction_ids"]),
            archive_transaction_id=value["archive_transaction_id"],
        )


@dataclass(frozen=True)
class FrontRuntimeState:
    front_id: str
    competition_state: Mapping[str, Any]
    candidate_ids: tuple[str, ...]
    lineage_rng_state: Mapping[str, Any]
    interval_count: int = 0
    mechanically_active_candidate_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        candidates = _sorted_ids(self.candidate_ids, "candidate_ids")
        active = _sorted_ids(
            candidates if self.mechanically_active_candidate_ids is None
            else self.mechanically_active_candidate_ids,
            "mechanically_active_candidate_ids",
        )
        if not set(active).issubset(candidates):
            raise ValueError("mechanically active candidates must belong to the competition")
        if int(self.interval_count) < 0:
            raise ValueError("front interval count must be nonnegative")
        object.__setattr__(self, "candidate_ids", candidates)
        object.__setattr__(self, "mechanically_active_candidate_ids", active)
        object.__setattr__(self, "competition_state", _mapping(self.competition_state))
        object.__setattr__(self, "lineage_rng_state", _mapping(self.lineage_rng_state))
        object.__setattr__(self, "interval_count", int(self.interval_count))

    def to_dict(self) -> dict[str, Any]:
        return {
            "front_id": self.front_id, "competition_state": dict(self.competition_state),
            "candidate_ids": list(self.candidate_ids),
            "mechanically_active_candidate_ids": list(
                self.mechanically_active_candidate_ids
            ),
            "dormant_candidate_ids": list(self.dormant_candidate_ids),
            "lineage_rng_state": dict(self.lineage_rng_state),
            "interval_count": self.interval_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrontRuntimeState":
        payload = dict(value)
        payload.pop("dormant_candidate_ids", None)
        return cls(**payload)

    @property
    def dormant_candidate_ids(self) -> tuple[str, ...]:
        active = set(self.mechanically_active_candidate_ids)
        return tuple(candidate for candidate in self.candidate_ids if candidate not in active)

    def activate_complete_competition(self) -> "FrontRuntimeState":
        return replace(self, mechanically_active_candidate_ids=self.candidate_ids)


@dataclass(frozen=True)
class GlobalSchedulerState:
    accepted_interval_index: int = 0
    accepted_transaction_index: int = 0
    tie_tolerance_s: float = 1.0e-12
    last_selection_key: str | None = None
    global_hazard_seed: int = 0
    competition_event_index: int = 0

    def __post_init__(self) -> None:
        if int(self.accepted_interval_index) < 0 or int(self.accepted_transaction_index) < 0:
            raise ValueError("scheduler indices must be nonnegative")
        tolerance = _finite_nonnegative(self.tie_tolerance_s, "tie_tolerance_s")
        object.__setattr__(self, "accepted_interval_index", int(self.accepted_interval_index))
        object.__setattr__(self, "accepted_transaction_index", int(self.accepted_transaction_index))
        object.__setattr__(self, "tie_tolerance_s", tolerance)
        object.__setattr__(self, "global_hazard_seed", int(self.global_hazard_seed))
        object.__setattr__(self, "competition_event_index", int(self.competition_event_index))

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_interval_index": self.accepted_interval_index,
            "accepted_transaction_index": self.accepted_transaction_index,
            "tie_tolerance_s": self.tie_tolerance_s,
            "last_selection_key": self.last_selection_key,
            "global_hazard_seed": self.global_hazard_seed,
            "competition_event_index": self.competition_event_index,
        }


@dataclass(frozen=True)
class ResourcePolicy:
    branching_mode: str = "mechanistic"
    front_resource_limit: int | None = None
    branch_transaction_limit: int | None = None
    shared_region_branching: str = "forbid"
    scheduler_policy: str = "v11_correlated_proposal_compatibility"

    def __post_init__(self) -> None:
        if self.branching_mode not in BRANCHING_MODES:
            raise ValueError(f"unsupported branching mode: {self.branching_mode}")
        if self.shared_region_branching not in SHARED_REGION_BRANCHING_MODES:
            raise ValueError(
                f"unsupported shared-region branching mode: {self.shared_region_branching}"
            )
        if self.scheduler_policy not in SCHEDULER_POLICIES:
            raise ValueError(f"unsupported scheduler policy: {self.scheduler_policy}")
        for name in ("front_resource_limit", "branch_transaction_limit"):
            value = getattr(self, name)
            if value is not None and int(value) < 1:
                raise ValueError(f"{name} must be positive or None")
            if value is not None:
                object.__setattr__(self, name, int(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "branching_mode": self.branching_mode,
            "front_resource_limit": self.front_resource_limit,
            "branch_transaction_limit": self.branch_transaction_limit,
            "shared_region_branching": self.shared_region_branching,
            "scheduler_policy": self.scheduler_policy,
        }


@dataclass(frozen=True)
class FrontCandidateObservation:
    accepted_state_id: str
    stress_field_state_id: str
    front_id: str
    owner_id: str
    candidate_id: str
    tip_coordinates_m: tuple[float, float]
    signed_local_J_J_per_m2: float
    marginal_G_J_per_m2: float
    kinetic_J_used_J_per_m2: float
    directional_K_MPa_sqrt_m: float
    directional_rate_per_s: float
    tensor: tuple[float, ...]
    tensor_reliability: str
    controlling_scalar_K_tip_id: str
    tensor_probe_tip_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "tip_coordinates_m", _finite_point(
            self.tip_coordinates_m, "tip_coordinates_m"
        ))
        for name in (
            "signed_local_J_J_per_m2", "marginal_G_J_per_m2",
            "kinetic_J_used_J_per_m2", "directional_K_MPa_sqrt_m",
            "directional_rate_per_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        tensor = tuple(float(item) for item in self.tensor)
        if not tensor or not all(math.isfinite(item) for item in tensor):
            raise ValueError("the local tensor must be finite and nonempty")
        object.__setattr__(self, "tensor", tensor)
        if not (
            self.front_id == self.controlling_scalar_K_tip_id == self.tensor_probe_tip_id
        ):
            raise ValueError("scalar K and tensor must belong to the same front")

    @property
    def identity_key(self) -> tuple[str, str, str, str]:
        return self.owner_id, self.front_id, self.candidate_id, self.accepted_state_id


@dataclass(frozen=True)
class TopologyProposal:
    proposal_id: str
    action_type: str
    front_id: str
    owner_id: str
    candidate_ids: tuple[str, ...]
    completion_time_s: float
    end_points_m: tuple[tuple[float, float], ...] = ()
    target_front_id: str | None = None
    member_event_ids: tuple[str, ...] = ()
    member_event_ordinals: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.action_type not in {"one_arm", "two_arm", "coalescence", "retirement"}:
            raise ValueError(f"unsupported scheduler proposal: {self.action_type}")
        raw_candidates = tuple(str(item) for item in self.candidate_ids)
        if len(raw_candidates) != len(set(raw_candidates)):
            raise ValueError("proposal candidate_ids contains duplicate identifiers")
        required = {"one_arm": 1, "two_arm": 2}.get(self.action_type)
        if required is not None and len(raw_candidates) != required:
            raise ValueError(f"{self.action_type} requires {required} candidate identities")
        if required is None and raw_candidates:
            raise ValueError(f"{self.action_type} must not consume cleavage candidates")
        raw_points = tuple(_finite_point(item, "proposal end point") for item in self.end_points_m)
        if required is not None and len(raw_points) != required:
            raise ValueError(f"{self.action_type} requires {required} realized end points")
        if required is not None:
            pairs = tuple(sorted(zip(raw_candidates, raw_points), key=lambda item: item[0]))
            candidates = tuple(item[0] for item in pairs)
            points = tuple(item[1] for item in pairs)
        else:
            candidates, points = (), raw_points
        object.__setattr__(self, "candidate_ids", candidates)
        object.__setattr__(self, "end_points_m", points)
        object.__setattr__(self, "completion_time_s", _finite_nonnegative(
            self.completion_time_s, "completion_time_s"
        ))
        event_ids = tuple(str(item) for item in self.member_event_ids)
        ordinals = tuple(int(item) for item in self.member_event_ordinals)
        if event_ids and len(event_ids) != len(candidates):
            raise ValueError("proposal event identities must map one-to-one to candidates")
        if ordinals and len(ordinals) != len(candidates):
            raise ValueError("proposal event ordinals must map one-to-one to candidates")
        if any(item <= 0 for item in ordinals):
            raise ValueError("proposal event ordinals must be positive")
        if event_ids:
            by_candidate = sorted(zip(candidates, event_ids, ordinals or (0,) * len(candidates)))
            object.__setattr__(self, "member_event_ids", tuple(item[1] for item in by_candidate))
            if ordinals:
                object.__setattr__(self, "member_event_ordinals", tuple(item[2] for item in by_candidate))

    @property
    def selection_key(self) -> str:
        return canonical_hash({
            "action_type": self.action_type, "candidate_ids": self.candidate_ids,
            "front_id": self.front_id, "owner_id": self.owner_id,
            "proposal_id": self.proposal_id,
            "member_event_ids": self.member_event_ids,
        })


@dataclass(frozen=True)
class TopologyTransactionRecord:
    transaction_id: str
    action_type: str
    selected_front_id: str
    selected_owner_id: str
    event_candidate_ids: tuple[str, ...]
    created_front_ids: tuple[str, ...]
    retired_front_ids: tuple[str, ...]
    created_junction_id: str | None
    pre_active_front_count: int
    post_active_front_count: int
    realized_lengths_m: tuple[float, ...]
    owner_region_transition: Mapping[str, Any]
    renewal_owner_id: str | None
    renewal_distance_m: float
    pre_topology_fingerprint: str
    post_topology_fingerprint: str
    pre_registry_fingerprint: str
    post_registry_fingerprint: str
    proposal_id: str = ""
    member_event_ids: tuple[str, ...] = ()
    member_event_ordinals: tuple[int, ...] = ()
    completion_times_s: tuple[float, ...] = ()
    realized_endpoints_m: tuple[tuple[float, float], ...] = ()
    coalescence_target_front_id: str | None = None
    competition_hashes_before: Mapping[str, str] | None = None
    competition_hashes_after: Mapping[str, str] | None = None
    rng_hashes_before: Mapping[str, str] | None = None
    rng_hashes_after: Mapping[str, str] | None = None
    accepted_state_id_before: str = ""
    accepted_state_id_after: str = ""
    exact_accepted_trial_fingerprint: str = ""
    clipped_or_coalesced_disposition: str | None = None
    wake_mutation: Mapping[str, Any] | None = None
    stored_energy_release_J_per_m: float = 0.0
    stored_energy_cost_J_per_m: float = 0.0
    accepted_fem_topology_fingerprint: str = ""
    geometry_fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.action_type not in TRANSACTION_KINDS:
            raise ValueError(f"unsupported transaction type: {self.action_type}")
        object.__setattr__(self, "event_candidate_ids", _sorted_ids(
            self.event_candidate_ids, "event_candidate_ids", allow_empty=True
        ))
        object.__setattr__(self, "created_front_ids", _sorted_ids(
            self.created_front_ids, "created_front_ids", allow_empty=True
        ))
        object.__setattr__(self, "retired_front_ids", _sorted_ids(
            self.retired_front_ids, "retired_front_ids", allow_empty=True
        ))
        lengths = tuple(_finite_nonnegative(item, "realized length") for item in self.realized_lengths_m)
        object.__setattr__(self, "realized_lengths_m", lengths)
        object.__setattr__(self, "renewal_distance_m", _finite_nonnegative(
            self.renewal_distance_m, "renewal_distance_m"
        ))
        object.__setattr__(self, "owner_region_transition", _mapping(self.owner_region_transition))
        object.__setattr__(self, "member_event_ids", tuple(str(x) for x in self.member_event_ids))
        object.__setattr__(self, "member_event_ordinals", tuple(int(x) for x in self.member_event_ordinals))
        object.__setattr__(self, "completion_times_s", tuple(
            _finite_nonnegative(x, "completion time") for x in self.completion_times_s
        ))
        object.__setattr__(self, "realized_endpoints_m", tuple(
            _finite_point(x, "realized endpoint") for x in self.realized_endpoints_m
        ))
        for name in (
            "competition_hashes_before", "competition_hashes_after",
            "rng_hashes_before", "rng_hashes_after",
        ):
            object.__setattr__(self, name, _mapping(getattr(self, name)))
        object.__setattr__(self, "wake_mutation", _mapping(self.wake_mutation))
        object.__setattr__(self, "stored_energy_release_J_per_m", _finite_nonnegative(
            self.stored_energy_release_J_per_m, "stored energy release"
        ))
        object.__setattr__(self, "stored_energy_cost_J_per_m", _finite_nonnegative(
            self.stored_energy_cost_J_per_m, "stored energy cost"
        ))

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.__dict__)
        for key in (
            "event_candidate_ids", "created_front_ids", "retired_front_ids",
            "realized_lengths_m", "member_event_ids", "member_event_ordinals",
            "completion_times_s", "realized_endpoints_m",
        ):
            result[key] = list(result[key])
        result["owner_region_transition"] = dict(self.owner_region_transition)
        result["wake_mutation"] = dict(self.wake_mutation or {})
        for key in (
            "competition_hashes_before", "competition_hashes_after",
            "rng_hashes_before", "rng_hashes_after",
        ):
            result[key] = dict(result[key] or {})
        return result

    @property
    def transaction_fingerprint(self) -> str:
        return canonical_hash(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TopologyTransactionRecord":
        return cls(**value)


@dataclass(frozen=True)
class CouplingEvidence:
    junction_id: str
    branch_handoff_length_m: float
    process_zone_length_m: float
    local_contour_radius_m: float
    actual_tip_separation_m: float
    minimum_post_junction_path_length_m: float
    contours_overlap: bool
    detached_front_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "branch_handoff_length_m", "process_zone_length_m", "local_contour_radius_m",
            "actual_tip_separation_m", "minimum_post_junction_path_length_m",
        ):
            object.__setattr__(self, name, _finite_nonnegative(getattr(self, name), name))
        object.__setattr__(self, "detached_front_ids", _sorted_ids(
            self.detached_front_ids, "detached_front_ids", allow_empty=True,
        ))

    @property
    def remains_coupled(self) -> bool:
        separation_scale = max(self.process_zone_length_m, 2.0 * self.local_contour_radius_m)
        return (
            self.minimum_post_junction_path_length_m < self.branch_handoff_length_m
            or self.actual_tip_separation_m < separation_scale
            or self.contours_overlap
        )


@dataclass(frozen=True)
class MultiFrontRuntimeState:
    crack_network: CrackNetworkState
    front_runtimes: Mapping[str, FrontRuntimeState]
    owner_by_front: Mapping[str, str]
    process_regions: Mapping[str, ProcessRegionState]
    process_engines: Mapping[str, ProcessEngineState]
    junctions: Mapping[str, BranchJunctionState]
    reservoirs: Mapping[str, ProcessRegionReservoir]
    scheduler: GlobalSchedulerState
    resource_policy: ResourcePolicy
    accepted_state_id: str
    stress_field_state_id: str
    compatibility_provenance: Mapping[str, Any] | None = None
    cumulative_branch_births: int = 0
    cumulative_coalescences: int = 0
    cumulative_retirements: int = 0
    transaction_records: tuple[TopologyTransactionRecord, ...] = ()
    output_counters: Mapping[str, int] | None = None
    termination_reason: str | None = None
    policy_bound: bool = False

    def __post_init__(self) -> None:
        for name in (
            "front_runtimes", "owner_by_front", "process_regions", "process_engines",
            "junctions", "reservoirs",
        ):
            object.__setattr__(self, name, dict(sorted(dict(getattr(self, name)).items())))
        for name in ("cumulative_branch_births", "cumulative_coalescences", "cumulative_retirements"):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "transaction_records", tuple(self.transaction_records))
        object.__setattr__(self, "compatibility_provenance", _mapping(
            self.compatibility_provenance
        ))
        counters = {str(key): int(item) for key, item in (self.output_counters or {}).items()}
        if any(item < 0 for item in counters.values()):
            raise ValueError("output counters must be nonnegative")
        object.__setattr__(self, "output_counters", dict(sorted(counters.items())))
        self.validate()

    @classmethod
    def one_front(
        cls, network: CrackNetworkState, front_runtime: FrontRuntimeState,
        engine: ProcessEngineState, *, resource_policy: ResourcePolicy,
        accepted_state_id: str, stress_field_state_id: str,
    ) -> "MultiFrontRuntimeState":
        active = network.active_tip_ids
        if len(active) != 1 or active[0] != front_runtime.front_id:
            raise ValueError("one-front initialization requires the unique active front")
        owner_id = f"owner:{front_runtime.front_id}"
        region = ProcessRegionState(
            owner_id=owner_id, member_front_ids=frozenset(active),
            unresolved_junction_ids=frozenset(),
            cumulative_process_advance_m=engine.local_process_coordinate_m,
            process_engine_id=engine.engine_id, source_state_id=engine.source_state_id,
        )
        return cls(
            crack_network=network, front_runtimes={front_runtime.front_id: front_runtime},
            owner_by_front={front_runtime.front_id: owner_id}, process_regions={owner_id: region},
            process_engines={engine.engine_id: engine}, junctions={}, reservoirs={},
            scheduler=GlobalSchedulerState(), resource_policy=resource_policy,
            accepted_state_id=accepted_state_id, stress_field_state_id=stress_field_state_id,
        )

    @property
    def active_front_ids(self) -> tuple[str, ...]:
        return self.crack_network.active_tip_ids

    @property
    def registry_fingerprint(self) -> str:
        return canonical_hash({
            "front_runtimes": {key: value.to_dict() for key, value in self.front_runtimes.items()},
            "owner_by_front": dict(self.owner_by_front),
            "process_regions": {key: value.to_dict() for key, value in self.process_regions.items()},
            "process_engines": {key: value.to_dict() for key, value in self.process_engines.items()},
            "junctions": {key: value.to_dict() for key, value in self.junctions.items()},
            "reservoirs": {key: value.to_dict() for key, value in self.reservoirs.items()},
        })

    @property
    def topology_fingerprint(self) -> str:
        return hashlib.sha256(self.crack_network.to_json().encode()).hexdigest()

    def validate(self) -> None:
        active = set(self.active_front_ids)
        if set(self.front_runtimes) != active:
            raise ValueError("every active front must have exactly one competition/runtime")
        if set(self.owner_by_front) != active:
            raise ValueError("every active front must have exactly one process owner")
        if any(runtime.front_id != key for key, runtime in self.front_runtimes.items()):
            raise ValueError("front runtime identity mismatch")
        covered: set[str] = set()
        engine_ids: list[str] = []
        for owner_id, region in self.process_regions.items():
            if region.owner_id != owner_id:
                raise ValueError("process-region identity mismatch")
            if covered.intersection(region.member_front_ids):
                raise ValueError("an active front belongs to more than one process region")
            covered.update(region.member_front_ids)
            if region.process_engine_id not in self.process_engines:
                raise ValueError("process region references a missing mutable engine")
            engine = self.process_engines[region.process_engine_id]
            if engine.source_state_id != region.source_state_id:
                raise ValueError("process region and engine source-state identities differ")
            if not math.isclose(
                engine.local_process_coordinate_m, region.cumulative_process_advance_m,
                rel_tol=0.0, abs_tol=1.0e-18,
            ):
                raise ValueError("owner-local process coordinate differs from its engine")
            engine_ids.append(region.process_engine_id)
            if not region.unresolved_junction_ids.issubset(self.junctions):
                raise ValueError("process region references a missing junction")
            if (
                region.source_reservoir_id is not None
                and region.source_reservoir_id not in self.reservoirs
            ):
                raise ValueError("fresh process region references a missing source reservoir")
        if covered != active:
            raise ValueError("process-region membership does not exactly cover active fronts")
        if len(engine_ids) != len(set(engine_ids)):
            raise ValueError("independent process owners alias one mutable engine")
        if set(self.process_engines) != set(engine_ids):
            raise ValueError("a mutable process engine has no unique active owner")
        if any(engine.engine_id != key for key, engine in self.process_engines.items()):
            raise ValueError("process-engine registry key differs from engine identity")
        if len({id(engine) for engine in self.process_engines.values()}) != len(self.process_engines):
            raise ValueError("independent owners alias the same process-engine object")
        for front_id, owner_id in self.owner_by_front.items():
            if owner_id not in self.process_regions or front_id not in self.process_regions[owner_id].member_front_ids:
                raise ValueError("front-to-owner registry is inconsistent with region membership")
        branches = {item.branch_id for item in self.crack_network.branches}
        for junction in self.junctions.values():
            if junction.parent_branch_id not in branches or not set(junction.child_branch_ids).issubset(branches):
                raise ValueError("junction references a missing crack branch")
            if junction.reservoir_id is not None and junction.reservoir_id not in self.reservoirs:
                raise ValueError("junction references a missing reservoir")
        if self.termination_reason in {POLICY_BOUND_FRONT_LIMIT, POLICY_BOUND_TRANSACTION_LIMIT} and not self.policy_bound:
            raise ValueError("a configured resource stop must be classified policy-bound")
        expected_active = (
            1 + self.cumulative_branch_births
            - self.cumulative_coalescences - self.cumulative_retirements
        )
        if expected_active != len(active):
            raise ValueError("active-front accounting does not close")

    def total_conserved_ledgers(self) -> dict[str, float]:
        active = [engine.conserved_ledgers for engine in self.process_engines.values()]
        archived = [item.archived_engine.conserved_ledgers for item in self.reservoirs.values()]
        return _ledger_sum(*(active + archived))

    def total_signed_system_ledgers(self) -> dict[str, float]:
        active = [engine.signed_system_ledgers for engine in self.process_engines.values()]
        archived = [
            item.archived_engine.signed_system_ledgers for item in self.reservoirs.values()
        ]
        return _ledger_sum(*(active + archived))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "crack_network": self.crack_network.to_dict(),
            "front_runtimes": {key: value.to_dict() for key, value in self.front_runtimes.items()},
            "owner_by_front": dict(self.owner_by_front),
            "process_regions": {key: value.to_dict() for key, value in self.process_regions.items()},
            "process_engines": {key: value.to_dict() for key, value in self.process_engines.items()},
            "junctions": {key: value.to_dict() for key, value in self.junctions.items()},
            "reservoirs": {key: value.to_dict() for key, value in self.reservoirs.items()},
            "scheduler": self.scheduler.to_dict(),
            "resource_policy": self.resource_policy.to_dict(),
            "accepted_state_id": self.accepted_state_id,
            "stress_field_state_id": self.stress_field_state_id,
            "compatibility_provenance": dict(self.compatibility_provenance or {}),
            "cumulative_branch_births": self.cumulative_branch_births,
            "cumulative_coalescences": self.cumulative_coalescences,
            "cumulative_retirements": self.cumulative_retirements,
            "transaction_records": [item.to_dict() for item in self.transaction_records],
            "output_counters": dict(self.output_counters or {}),
            "termination_reason": self.termination_reason,
            "policy_bound": self.policy_bound,
            "topology_fingerprint": self.topology_fingerprint,
            "registry_fingerprint": self.registry_fingerprint,
            "conserved_ledgers": self.total_conserved_ledgers(),
            "signed_system_ledgers": self.total_signed_system_ledgers(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MultiFrontRuntimeState":
        if value.get("schema") != SCHEMA:
            raise ValueError("unsupported V12 multi-front runtime schema")
        result = cls(
            crack_network=CrackNetworkState.from_dict(value["crack_network"]),
            front_runtimes={key: FrontRuntimeState.from_dict(item) for key, item in value["front_runtimes"].items()},
            owner_by_front=value["owner_by_front"],
            process_regions={key: ProcessRegionState.from_dict(item) for key, item in value["process_regions"].items()},
            process_engines={key: ProcessEngineState.from_dict(item) for key, item in value["process_engines"].items()},
            junctions={key: BranchJunctionState.from_dict(item) for key, item in value["junctions"].items()},
            reservoirs={key: ProcessRegionReservoir.from_dict(item) for key, item in value["reservoirs"].items()},
            scheduler=GlobalSchedulerState(**value["scheduler"]),
            resource_policy=ResourcePolicy(**value["resource_policy"]),
            accepted_state_id=value["accepted_state_id"], stress_field_state_id=value["stress_field_state_id"],
            compatibility_provenance=value.get("compatibility_provenance", {}),
            cumulative_branch_births=value["cumulative_branch_births"],
            cumulative_coalescences=value["cumulative_coalescences"],
            cumulative_retirements=value["cumulative_retirements"],
            transaction_records=tuple(TopologyTransactionRecord.from_dict(item) for item in value["transaction_records"]),
            output_counters=value.get("output_counters", {}),
            termination_reason=value.get("termination_reason"), policy_bound=value.get("policy_bound", False),
        )
        sentinels = {
            "topology_fingerprint": result.topology_fingerprint,
            "registry_fingerprint": result.registry_fingerprint,
            "conserved_ledgers": result.total_conserved_ledgers(),
            "signed_system_ledgers": result.total_signed_system_ledgers(),
        }
        for key, expected in sentinels.items():
            if value.get(key) != expected:
                raise ValueError(f"V12 derived checkpoint sentinel differs: {key}")
        return result


@dataclass(frozen=True)
class AcceptedIntervalResult:
    state: MultiFrontRuntimeState
    controlling_observation_by_owner: Mapping[str, FrontCandidateObservation]
    selected_proposal: TopologyProposal | None


def _global_proposal_choice(
    proposals: Sequence[TopologyProposal], tolerance_s: float, *,
    scheduler_policy: str = "v12_global_earliest_proposal",
    global_hazard_seed: int = 0,
    competition_event_index: int = 0,
) -> TopologyProposal | None:
    if not proposals:
        return None
    earliest = min(item.completion_time_s for item in proposals)
    tied = [item for item in proposals if item.completion_time_s <= earliest + tolerance_s]
    if scheduler_policy == "v11_correlated_proposal_compatibility":
        return min(tied, key=lambda item: canonical_hash({
            "global_hazard_seed": int(global_hazard_seed),
            "competition_event_index": int(competition_event_index),
            "front_id": item.front_id,
            "member_candidate_ids": item.candidate_ids,
            "member_event_ids": item.member_event_ids,
        }))
    if scheduler_policy == "v12_global_earliest_proposal":
        return min(tied, key=lambda item: item.selection_key)
    raise ValueError(f"unsupported scheduler policy: {scheduler_policy}")


def form_correlated_topology_proposals(
    *, front_id: str, owner_id: str, action_proposals: Sequence[Any],
    end_points_by_candidate: Mapping[str, tuple[float, float]],
) -> tuple[TopologyProposal, ...]:
    """Convert complete same-tip V11 proposals before any global comparison.

    A candidate event belonging to an available correlated pair is not also
    exposed as a singleton.  This preserves the qualified atomic branch birth
    when the two clocks complete within the correlation interval.
    """
    values = tuple(action_proposals)
    paired_event_ids = {
        event_id
        for item in values if item.action_type == "two_arm"
        for event_id in item.member_event_ids
    }
    complete = tuple(
        item for item in values
        if item.action_type == "two_arm"
        or not paired_event_ids.intersection(item.member_event_ids)
    )
    result = []
    for item in complete:
        candidates = tuple(item.member_candidate_ids)
        result.append(TopologyProposal(
            proposal_id=item.action_id,
            action_type=item.action_type,
            front_id=front_id,
            owner_id=owner_id,
            candidate_ids=candidates,
            completion_time_s=max(float(value) for value in item.completion_times_s),
            end_points_m=tuple(end_points_by_candidate[candidate] for candidate in candidates),
            member_event_ids=tuple(item.member_event_ids),
            member_event_ordinals=tuple(item.member_event_ordinals),
        ))
    return tuple(sorted(result, key=lambda item: item.selection_key))


def select_global_topology_proposal(
    proposals: Sequence[TopologyProposal], scheduler: GlobalSchedulerState, *,
    scheduler_policy: str,
) -> TopologyProposal | None:
    """Select globally between already-complete, same-tip topology proposals."""
    return _global_proposal_choice(
        proposals, scheduler.tie_tolerance_s,
        scheduler_policy=scheduler_policy,
        global_hazard_seed=scheduler.global_hazard_seed,
        competition_event_index=scheduler.competition_event_index,
    )


EngineEvolver = Callable[[ProcessEngineState, FrontCandidateObservation, float], ProcessEngineState]
FrontClockAdvancer = Callable[[FrontRuntimeState, Sequence[FrontCandidateObservation], float], FrontRuntimeState]


def advance_accepted_interval(
    state: MultiFrontRuntimeState,
    observations: Sequence[FrontCandidateObservation],
    proposals: Sequence[TopologyProposal],
    *,
    duration_s: float,
    evolve_engine: EngineEvolver | None = None,
    advance_front_clock: FrontClockAdvancer | None = None,
) -> AcceptedIntervalResult:
    """Evolve every owner once and select the earliest proposal over all fronts."""
    duration = _finite_nonnegative(duration_s, "accepted interval duration")
    if duration <= 0.0:
        raise ValueError("accepted interval duration must be positive")
    by_identity: dict[tuple[str, str], FrontCandidateObservation] = {}
    for observation in observations:
        if observation.accepted_state_id != state.accepted_state_id:
            raise ValueError("observation is not bound to the accepted state")
        if observation.stress_field_state_id != state.stress_field_state_id:
            raise ValueError("observation is not bound to the accepted stress state")
        expected_owner = state.owner_by_front.get(observation.front_id)
        if expected_owner != observation.owner_id:
            raise ValueError("observation process-owner identity differs from the registry")
        key = observation.front_id, observation.candidate_id
        if key in by_identity:
            raise ValueError("duplicate front/candidate observation")
        by_identity[key] = observation
    expected = {
        (front_id, candidate_id)
        for front_id, runtime in state.front_runtimes.items()
        for candidate_id in runtime.mechanically_active_candidate_ids
    }
    if set(by_identity) != expected:
        raise ValueError(
            "observations must cover every mechanically active front/candidate exactly once"
        )

    controlling: dict[str, FrontCandidateObservation] = {}
    engines = dict(state.process_engines)
    for owner_id, region in state.process_regions.items():
        pool = [
            observation for observation in observations
            if observation.front_id in region.member_front_ids
        ]
        selected = min(
            pool,
            key=lambda item: (-item.directional_K_MPa_sqrt_m, item.identity_key),
        )
        controlling[owner_id] = selected
        before = engines[region.process_engine_id]
        after = (
            before.evolve_once() if evolve_engine is None
            else evolve_engine(before, selected, duration)
        )
        if after.engine_id != before.engine_id or after.update_count != before.update_count + 1:
            raise ValueError("each process owner must evolve its own engine exactly once")
        if after.event_renewal_count != before.event_renewal_count:
            raise ValueError("time evolution must not perform topology-owned renewal")
        engines[region.process_engine_id] = after

    front_runtimes = {}
    for front_id, runtime in state.front_runtimes.items():
        local = [item for item in observations if item.front_id == front_id]
        after = (
            replace(runtime, interval_count=runtime.interval_count + 1)
            if advance_front_clock is None
            else advance_front_clock(runtime, local, duration)
        )
        if after.front_id != front_id or after.interval_count != runtime.interval_count + 1:
            raise ValueError("each active front clock must advance exactly once")
        front_runtimes[front_id] = after

    for proposal in proposals:
        if proposal.front_id not in state.front_runtimes:
            raise ValueError("proposal references an inactive front")
        if state.owner_by_front[proposal.front_id] != proposal.owner_id:
            raise ValueError("proposal owner differs from front registry")
    selected_proposal = select_global_topology_proposal(
        proposals, state.scheduler,
        scheduler_policy=state.resource_policy.scheduler_policy,
    )
    scheduler = replace(
        state.scheduler,
        accepted_interval_index=state.scheduler.accepted_interval_index + 1,
        last_selection_key=(None if selected_proposal is None else selected_proposal.selection_key),
    )
    evolved = replace(
        state, front_runtimes=front_runtimes, process_engines=engines,
        scheduler=scheduler,
    )
    return AcceptedIntervalResult(evolved, dict(sorted(controlling.items())), selected_proposal)


def _child_id(parent_id: str, candidate_id: str, transaction_index: int) -> str:
    return "b" + hashlib.sha256(
        f"{parent_id}|{candidate_id}|{transaction_index}".encode()
    ).hexdigest()[:15]


def _junction_id(
    parent_id: str, candidate_ids: Sequence[str], transaction_index: int,
) -> str:
    return "j" + hashlib.sha256(
        f"{parent_id}|{transaction_index}|{'|'.join(sorted(candidate_ids))}".encode()
    ).hexdigest()[:15]


def _replace_branch(network: CrackNetworkState, branch: CrackBranchState) -> CrackNetworkState:
    return CrackNetworkState(
        branches=tuple(branch if item.branch_id == branch.branch_id else item for item in network.branches),
        primary_branch_id=network.primary_branch_id,
        geometry_generation=network.geometry_generation + 1,
        branching_enabled=network.branching_enabled,
    )


def _with_renewal(
    regions: dict[str, ProcessRegionState], engines: dict[str, ProcessEngineState],
    owner_id: str, distance_m: float,
) -> None:
    region = regions[owner_id]
    engine = engines[region.process_engine_id].renew(distance_m)
    engines[region.process_engine_id] = engine
    regions[owner_id] = replace(
        region, cumulative_process_advance_m=engine.local_process_coordinate_m
    )


def _record(
    before: MultiFrontRuntimeState, after: MultiFrontRuntimeState, *,
    transaction_id: str, action_type: str, selected_front_id: str,
    selected_owner_id: str, event_candidate_ids: Sequence[str] = (),
    created_front_ids: Sequence[str] = (), retired_front_ids: Sequence[str] = (),
    created_junction_id: str | None = None, realized_lengths_m: Sequence[float] = (),
    owner_region_transition: Mapping[str, Any] | None = None,
    renewal_owner_id: str | None = None, renewal_distance_m: float = 0.0,
    proposal: TopologyProposal | None = None,
    exact_accepted_trial_fingerprint: str | None = None,
) -> MultiFrontRuntimeState:
    competition_before = {
        key: canonical_hash(value.competition_state)
        for key, value in before.front_runtimes.items()
    }
    competition_after = {
        key: canonical_hash(value.competition_state)
        for key, value in after.front_runtimes.items()
    }
    rng_before = {
        key: canonical_hash(value.lineage_rng_state)
        for key, value in before.front_runtimes.items()
    }
    rng_after = {
        key: canonical_hash(value.lineage_rng_state)
        for key, value in after.front_runtimes.items()
    }
    record = TopologyTransactionRecord(
        transaction_id=transaction_id, action_type=action_type,
        selected_front_id=selected_front_id, selected_owner_id=selected_owner_id,
        event_candidate_ids=tuple(event_candidate_ids), created_front_ids=tuple(created_front_ids),
        retired_front_ids=tuple(retired_front_ids), created_junction_id=created_junction_id,
        pre_active_front_count=len(before.active_front_ids),
        post_active_front_count=len(after.active_front_ids),
        realized_lengths_m=tuple(realized_lengths_m),
        owner_region_transition=dict(owner_region_transition or {}),
        renewal_owner_id=renewal_owner_id, renewal_distance_m=renewal_distance_m,
        pre_topology_fingerprint=before.topology_fingerprint,
        post_topology_fingerprint=after.topology_fingerprint,
        pre_registry_fingerprint=before.registry_fingerprint,
        post_registry_fingerprint=after.registry_fingerprint,
        proposal_id=("" if proposal is None else proposal.proposal_id),
        member_event_ids=(() if proposal is None else proposal.member_event_ids),
        member_event_ordinals=(() if proposal is None else proposal.member_event_ordinals),
        completion_times_s=(
            () if proposal is None else (proposal.completion_time_s,) * len(proposal.candidate_ids)
        ),
        realized_endpoints_m=(() if proposal is None else proposal.end_points_m),
        coalescence_target_front_id=(None if proposal is None else proposal.target_front_id),
        competition_hashes_before=competition_before,
        competition_hashes_after=competition_after,
        rng_hashes_before=rng_before, rng_hashes_after=rng_after,
        accepted_state_id_before=before.accepted_state_id,
        accepted_state_id_after=after.accepted_state_id,
        exact_accepted_trial_fingerprint=(
            after.topology_fingerprint
            if exact_accepted_trial_fingerprint is None
            else exact_accepted_trial_fingerprint
        ),
    )
    scheduler = replace(
        after.scheduler,
        accepted_transaction_index=after.scheduler.accepted_transaction_index + 1,
    )
    return replace(after, transaction_records=before.transaction_records + (record,), scheduler=scheduler)


def extend_one_arm(state: MultiFrontRuntimeState, proposal: TopologyProposal) -> MultiFrontRuntimeState:
    if proposal.action_type != "one_arm":
        raise ValueError("one-arm extension requires a one_arm proposal")
    branch = state.crack_network.branch(proposal.front_id)
    end = proposal.end_points_m[0]
    length = math.dist(branch.tip, end)
    angle = math.atan2(end[1] - branch.tip[1], end[0] - branch.tip[0])
    orientations = (
        (angle,) if len(branch.path) == 1
        else branch.orientation_history_rad + (angle,)
    )
    updated = replace(
        branch, path=branch.path + (end,),
        orientation_history_rad=orientations,
    )
    network = _replace_branch(state.crack_network, updated)
    regions, engines = dict(state.process_regions), dict(state.process_engines)
    _with_renewal(regions, engines, proposal.owner_id, length)
    provisional = replace(state, crack_network=network, process_regions=regions, process_engines=engines)
    transaction_id = f"tx:{state.scheduler.accepted_transaction_index + 1:08d}"
    return _record(
        state, provisional, transaction_id=transaction_id, action_type="one_arm",
        selected_front_id=proposal.front_id, selected_owner_id=proposal.owner_id,
        event_candidate_ids=proposal.candidate_ids, realized_lengths_m=(length,),
        renewal_owner_id=proposal.owner_id, renewal_distance_m=length,
        proposal=proposal,
    )


def bifurcate_binary(state: MultiFrontRuntimeState, proposal: TopologyProposal) -> MultiFrontRuntimeState:
    if proposal.action_type != "two_arm":
        raise ValueError("binary bifurcation requires a two_arm proposal")
    if state.resource_policy.branching_mode != "mechanistic":
        raise ValueError("branching is disabled for this trajectory")
    parent = state.crack_network.branch(proposal.front_id)
    if parent.status != "active":
        raise ValueError("binary parent must be active")
    owner_id = state.owner_by_front[parent.branch_id]
    region = state.process_regions[owner_id]
    if (
        state.resource_policy.shared_region_branching == "forbid"
        and (
            len(region.member_front_ids) != 1
            or bool(region.unresolved_junction_ids)
        )
    ):
        raise ValueError(PARENT_PROCESS_ZONE_UNRESOLVED)
    post_count = len(state.active_front_ids) + 1
    if (
        state.resource_policy.front_resource_limit is not None
        and post_count > state.resource_policy.front_resource_limit
    ):
        return replace(
            state, termination_reason=POLICY_BOUND_FRONT_LIMIT, policy_bound=True,
        )
    if (
        state.resource_policy.branch_transaction_limit is not None
        and state.cumulative_branch_births >= state.resource_policy.branch_transaction_limit
    ):
        return replace(
            state, termination_reason=POLICY_BOUND_TRANSACTION_LIMIT, policy_bound=True,
        )
    index = state.scheduler.accepted_transaction_index + 1
    child_ids = tuple(
        _child_id(parent.branch_id, candidate, index)
        for candidate in proposal.candidate_ids
    )
    children = []
    lengths = []
    for child_id, candidate, end in zip(child_ids, proposal.candidate_ids, proposal.end_points_m):
        length = math.dist(parent.tip, end)
        if length <= 0.0:
            raise ValueError("binary daughter must have positive realized length")
        lengths.append(length)
        children.append(CrackBranchState(
            branch_id=child_id, parent_branch_id=parent.branch_id,
            generation=parent.generation + 1, initiation_event=index,
            path=(parent.tip,),
            orientation_history_rad=(parent.current_orientation_rad,),
            local_state={"candidate_id": candidate, "cluster_unresolved": True},
        ))
    network = CrackNetworkState(
        branches=tuple(
            replace(item, status="terminated") if item.branch_id == parent.branch_id else item
            for item in state.crack_network.branches
        ) + tuple(children),
        primary_branch_id=state.crack_network.primary_branch_id,
        geometry_generation=state.crack_network.geometry_generation,
        branching_enabled=True,
    )
    # Reuse the V11 realized-network contract exactly: cluster birth itself
    # does not advance the geometry generation, and each accepted daughter arm
    # is then appended as one committed edge.
    from .topology_transaction_v11 import TopologyArm, extend_network_arm
    for child_id, candidate, end, length in zip(
        child_ids, proposal.candidate_ids, proposal.end_points_m, lengths,
    ):
        network = extend_network_arm(network, TopologyArm(
            candidate, child_id, parent.tip, end, length, 0.0,
        ))
    transaction_id = f"tx:{index:08d}"
    junction_id = _junction_id(parent.branch_id, proposal.candidate_ids, index)
    junction = BranchJunctionState(
        junction_id=junction_id, parent_branch_id=parent.branch_id,
        child_branch_ids=child_ids, birth_transaction_id=transaction_id,
        junction_xy_m=parent.tip,
    )
    regions = dict(state.process_regions)
    regions[owner_id] = replace(
        region,
        member_front_ids=(region.member_front_ids - {parent.branch_id}) | set(child_ids),
        unresolved_junction_ids=region.unresolved_junction_ids | {junction_id},
    )
    owner_by_front = dict(state.owner_by_front); owner_by_front.pop(parent.branch_id)
    front_runtimes = dict(state.front_runtimes); parent_runtime = front_runtimes.pop(parent.branch_id)
    for child_id, candidate in zip(child_ids, proposal.candidate_ids):
        seed = canonical_hash({
            "parent_rng": parent_runtime.lineage_rng_state,
            "junction_id": junction_id, "child_id": child_id,
        })
        front_runtimes[child_id] = FrontRuntimeState(
            # V11 assigns the complete accepted post-action competition value
            # to every active tip.  Retain that value; never replace it with a
            # lineage marker that cannot be resumed.
            front_id=child_id, competition_state=parent_runtime.competition_state,
            candidate_ids=parent_runtime.candidate_ids,
            lineage_rng_state={
                **dict(parent_runtime.lineage_rng_state),
                "deterministic_lineage_sha256": seed,
                "v11_lineage_parent_front_id": parent.branch_id,
                "v11_lineage_rule": "accepted_competition_value_for_every_active_tip",
            },
            interval_count=parent_runtime.interval_count,
            mechanically_active_candidate_ids=(candidate,),
        )
        owner_by_front[child_id] = owner_id
    engines = dict(state.process_engines)
    renewal = max(lengths)
    _with_renewal(regions, engines, owner_id, renewal)
    junctions = dict(state.junctions); junctions[junction_id] = junction
    provisional = replace(
        state, crack_network=network, front_runtimes=front_runtimes,
        owner_by_front=owner_by_front, process_regions=regions,
        process_engines=engines, junctions=junctions,
        cumulative_branch_births=state.cumulative_branch_births + 1,
    )
    return _record(
        state, provisional, transaction_id=transaction_id, action_type="two_arm",
        selected_front_id=parent.branch_id, selected_owner_id=owner_id,
        event_candidate_ids=proposal.candidate_ids, created_front_ids=child_ids,
        retired_front_ids=(parent.branch_id,), created_junction_id=junction_id,
        realized_lengths_m=lengths,
        owner_region_transition={"retained_owner_id": owner_id, "retained_engine_id": region.process_engine_id},
        renewal_owner_id=owner_id, renewal_distance_m=renewal,
        proposal=proposal,
    )


def _remove_front(
    state: MultiFrontRuntimeState, front_id: str, *, status: str, action_type: str,
    selected_owner_id: str, target_front_id: str | None = None,
    proposal: TopologyProposal | None = None,
) -> MultiFrontRuntimeState:
    branch = state.crack_network.branch(front_id)
    network = _replace_branch(state.crack_network, replace(branch, status=status))
    front_runtimes = dict(state.front_runtimes); front_runtimes.pop(front_id)
    owner_by_front = dict(state.owner_by_front); owner_id = owner_by_front.pop(front_id)
    regions, engines, reservoirs = (
        dict(state.process_regions), dict(state.process_engines), dict(state.reservoirs)
    )
    region = regions[owner_id]
    remaining = region.member_front_ids - {front_id}
    transition: dict[str, Any] = {"owner_id": owner_id, "removed_front_id": front_id}
    if remaining:
        regions[owner_id] = replace(region, member_front_ids=remaining)
    else:
        engine = engines.pop(region.process_engine_id); regions.pop(owner_id)
        tx_id = f"tx:{state.scheduler.accepted_transaction_index + 1:08d}"
        reservoir_id = f"reservoir:{canonical_hash({'owner': owner_id, 'tx': tx_id})[:20]}"
        reservoirs[reservoir_id] = ProcessRegionReservoir(
            reservoir_id=reservoir_id, archived_owner_id=owner_id,
            archived_engine=engine, member_front_ids_at_archive=(front_id,),
            junction_ids=tuple(region.unresolved_junction_ids), archive_transaction_id=tx_id,
        )
        transition["archived_reservoir_id"] = reservoir_id
    count_field = (
        "cumulative_coalescences" if action_type == "coalescence"
        else "cumulative_retirements"
    )
    provisional = replace(
        state, crack_network=network, front_runtimes=front_runtimes,
        owner_by_front=owner_by_front, process_regions=regions,
        process_engines=engines, reservoirs=reservoirs,
        **{count_field: getattr(state, count_field) + 1},
    )
    tx_id = f"tx:{state.scheduler.accepted_transaction_index + 1:08d}"
    transition["target_front_id"] = target_front_id
    return _record(
        state, provisional, transaction_id=tx_id, action_type=action_type,
        selected_front_id=front_id, selected_owner_id=selected_owner_id,
        retired_front_ids=(front_id,), owner_region_transition=transition,
        proposal=proposal,
    )


def retire_front(state: MultiFrontRuntimeState, proposal: TopologyProposal) -> MultiFrontRuntimeState:
    if proposal.action_type != "retirement":
        raise ValueError("front retirement requires a retirement proposal")
    return _remove_front(
        state, proposal.front_id, status="terminated", action_type="retirement",
        selected_owner_id=proposal.owner_id, proposal=proposal,
    )


def coalesce_front(state: MultiFrontRuntimeState, proposal: TopologyProposal) -> MultiFrontRuntimeState:
    if proposal.action_type != "coalescence" or proposal.target_front_id is None:
        raise ValueError("coalescence requires an incoming and target front")
    if proposal.target_front_id not in state.active_front_ids:
        raise ValueError("coalescence target must be active")
    return _remove_front(
        state, proposal.front_id, status="merged", action_type="coalescence",
        selected_owner_id=proposal.owner_id, target_front_id=proposal.target_front_id,
        proposal=proposal,
    )


def commit_selected_proposal(
    state: MultiFrontRuntimeState, proposal: TopologyProposal,
) -> MultiFrontRuntimeState:
    if state.termination_reason is not None:
        raise ValueError("cannot commit topology after trajectory termination")
    if state.owner_by_front.get(proposal.front_id) != proposal.owner_id:
        raise ValueError("selected proposal owner differs from runtime registry")
    return {
        "one_arm": extend_one_arm,
        "two_arm": bifurcate_binary,
        "coalescence": coalesce_front,
        "retirement": retire_front,
    }[proposal.action_type](state, proposal)


def reject_proposal(state: MultiFrontRuntimeState, _proposal: TopologyProposal) -> MultiFrontRuntimeState:
    """A rejected isolated trial returns the accepted state object unchanged."""
    return state


def _descendant_active_fronts(
    network: CrackNetworkState, branch_id: str, allowed: set[str],
) -> set[str]:
    by_id = {item.branch_id: item for item in network.branches}
    result = set()
    for front_id in allowed:
        cursor = by_id[front_id]
        while True:
            if cursor.branch_id == branch_id:
                result.add(front_id); break
            if cursor.parent_branch_id is None:
                break
            cursor = by_id[cursor.parent_branch_id]
    return result


def _components(nodes: Iterable[str], edges: Iterable[tuple[str, str]]) -> tuple[frozenset[str], ...]:
    adjacency = {node: set() for node in nodes}
    for left, right in edges:
        adjacency[left].add(right); adjacency[right].add(left)
    result = []
    unseen = set(adjacency)
    while unseen:
        root = min(unseen); stack = [root]; component = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node); unseen.discard(node)
            stack.extend(sorted(adjacency[node] - component, reverse=True))
        result.append(frozenset(component))
    return tuple(sorted(result, key=lambda item: tuple(sorted(item))))


def recompute_process_region_connectivity(
    state: MultiFrontRuntimeState,
    evidence_by_junction: Mapping[str, CouplingEvidence],
    *, fresh_engine_factory: Callable[
        [str, str, ProcessRegionState, frozenset[str]], ProcessEngineState
    ] | None = None,
) -> MultiFrontRuntimeState:
    """Resolve process ownership without dividing or copying historical state.

    A residual component with an unresolved junction retains the complete old
    owner, engine, coordinate, and ledger.  Detached components receive fresh
    engines.  The old engine is archived only after no unresolved shared
    component remains.
    """
    result = state
    for owner_id in tuple(sorted(state.process_regions)):
        region = result.process_regions.get(owner_id)
        if region is None or not region.unresolved_junction_ids:
            continue
        missing = region.unresolved_junction_ids - set(evidence_by_junction)
        if missing:
            raise ValueError(f"missing coupling evidence for junctions: {sorted(missing)}")
        edges = []
        resolved_ids = []
        for junction_id in sorted(region.unresolved_junction_ids):
            junction = result.junctions[junction_id]
            evidence = evidence_by_junction[junction_id]
            if evidence.junction_id != junction_id:
                raise ValueError("coupling evidence identity mismatch")
            detached = set(evidence.detached_front_ids)
            if not detached.issubset(region.member_front_ids):
                raise ValueError("detached front evidence lies outside its process owner")
            if evidence.remains_coupled:
                left = _descendant_active_fronts(
                    result.crack_network, junction.child_branch_ids[0], set(region.member_front_ids)
                ) - detached
                right = _descendant_active_fronts(
                    result.crack_network, junction.child_branch_ids[1], set(region.member_front_ids)
                ) - detached
                edges.extend((a, b) for a in left for b in right)
            else:
                resolved_ids.append(junction_id)
        components = _components(region.member_front_ids, edges)
        unresolved_by_component: list[frozenset[str]] = []
        for component in components:
            component_unresolved = set()
            for junction_id in region.unresolved_junction_ids:
                local_evidence = evidence_by_junction[junction_id]
                if not local_evidence.remains_coupled:
                    continue
                detached = set(local_evidence.detached_front_ids)
                left = _descendant_active_fronts(
                    result.crack_network,
                    result.junctions[junction_id].child_branch_ids[0],
                    set(region.member_front_ids),
                ) - detached
                right = _descendant_active_fronts(
                    result.crack_network,
                    result.junctions[junction_id].child_branch_ids[1],
                    set(region.member_front_ids),
                ) - detached
                local = set(component)
                if (local.intersection(left) and local.intersection(right)) or (
                    detached and local.intersection(left | right)
                ):
                    component_unresolved.add(junction_id)
            unresolved_by_component.append(frozenset(component_unresolved))
        represented_unresolved = set().union(*unresolved_by_component)
        resolved_ids.extend(
            junction_id
            for junction_id in region.unresolved_junction_ids
            if junction_id not in represented_unresolved
            and junction_id not in resolved_ids
        )
        residual_indexes = [
            index for index, values in enumerate(unresolved_by_component) if values
        ]
        if len(residual_indexes) > 1:
            raise ValueError(
                "multiple_residual_shared_components_require_regional_process_model"
            )
        junctions = dict(result.junctions)
        if len(components) == 1 and residual_indexes:
            for junction_id in resolved_ids:
                junctions[junction_id] = replace(junctions[junction_id], status="resolved")
            regions = dict(result.process_regions)
            regions[owner_id] = replace(
                region,
                unresolved_junction_ids=unresolved_by_component[0],
            )
            provisional = replace(result, junctions=junctions, process_regions=regions)
            if resolved_ids:
                tx_id = f"tx:{result.scheduler.accepted_transaction_index + 1:08d}"
                result = _record(
                    result, provisional, transaction_id=tx_id,
                    action_type="junction_resolution",
                    selected_front_id=min(region.member_front_ids),
                    selected_owner_id=owner_id,
                    owner_region_transition={
                        "resolved_junction_ids": sorted(resolved_ids),
                        "process_region_partitioned": False,
                    },
                )
            else:
                result = provisional
            continue

        tx_index = result.scheduler.accepted_transaction_index + 1
        tx_id = f"tx:{tx_index:08d}"
        engines = dict(result.process_engines)
        old_engine = engines[region.process_engine_id]
        regions = dict(result.process_regions); regions.pop(owner_id)
        reservoirs = dict(result.reservoirs)
        complete_handoff = not residual_indexes
        reservoir_id = None
        if complete_handoff:
            engines.pop(region.process_engine_id)
            reservoir_id = f"reservoir:{canonical_hash({'owner': owner_id, 'tx': tx_id})[:20]}"
            reservoirs[reservoir_id] = ProcessRegionReservoir(
                reservoir_id=reservoir_id, archived_owner_id=owner_id,
                archived_engine=old_engine,
                member_front_ids_at_archive=tuple(region.member_front_ids),
                junction_ids=tuple(region.unresolved_junction_ids), archive_transaction_id=tx_id,
            )
        owner_by_front = dict(result.owner_by_front)
        front_runtimes = dict(result.front_runtimes)
        fresh_owner_ids = []
        retained_component = None if complete_handoff else components[residual_indexes[0]]
        for component_index, component in enumerate(components):
            if component == retained_component:
                regions[owner_id] = replace(
                    region,
                    member_front_ids=component,
                    unresolved_junction_ids=unresolved_by_component[component_index],
                )
                for front_id in component:
                    owner_by_front[front_id] = owner_id
                continue
            component_key = canonical_hash({"old_owner": owner_id, "members": sorted(component), "tx": tx_id})[:20]
            new_owner = f"owner:{component_key}"; engine_id = f"engine:{component_key}"
            if fresh_engine_factory is None:
                if old_engine.checkpoint_payload_b64 is not None:
                    raise RuntimeError(
                        "production process handoff requires a complete fresh physical engine"
                    )
                engine = ProcessEngineState(
                    engine_id=engine_id, source_state_id=region.source_state_id,
                    active_ledgers={}, wake_ledgers={}, signed_system_ledgers={},
                    opaque_state_fingerprint="fresh_independent_engine_after_handoff",
                    mutable_state={"historical_state_imported": False},
                    rng_state={"historical_rng_imported": False},
                )
            else:
                engine = fresh_engine_factory(
                    new_owner, engine_id, region, component
                )
                if engine.engine_id != engine_id:
                    raise RuntimeError("fresh physical engine identity differs from handoff registry")
                if engine.local_process_coordinate_m != 0.0:
                    raise RuntimeError("fresh independent engine must begin at local coordinate zero")
                if engine.checkpoint_payload_b64 is None:
                    raise RuntimeError("fresh physical engine lacks a complete V11 checkpoint payload")
            engines[engine_id] = engine
            regions[new_owner] = ProcessRegionState(
                owner_id=new_owner, member_front_ids=component,
                unresolved_junction_ids=unresolved_by_component[component_index],
                cumulative_process_advance_m=0.0, process_engine_id=engine_id,
                source_state_id=engine.source_state_id,
                generation=region.generation + 1,
                source_reservoir_id=reservoir_id,
                detached_from_owner_id=owner_id,
            )
            for front_id in component:
                owner_by_front[front_id] = new_owner
                front_runtimes[front_id] = front_runtimes[
                    front_id
                ].activate_complete_competition()
            fresh_owner_ids.append(new_owner)
        for junction_id in resolved_ids:
            junctions[junction_id] = replace(
                junctions[junction_id], status="resolved", reservoir_id=reservoir_id,
            )
        provisional = replace(
            result, process_engines=engines, process_regions=regions,
            owner_by_front=owner_by_front, front_runtimes=front_runtimes,
            reservoirs=reservoirs, junctions=junctions,
        )
        result = _record(
            result, provisional, transaction_id=tx_id,
            action_type="process_region_partition", selected_front_id=min(region.member_front_ids),
            selected_owner_id=owner_id,
            owner_region_transition={
                "handoff_kind": (
                    "complete_archive_and_fresh_handoff"
                    if complete_handoff else "asymmetric_partial_handoff"
                ),
                "retained_owner_id": None if complete_handoff else owner_id,
                "retained_engine_id": None if complete_handoff else old_engine.engine_id,
                "retained_component_front_ids": (
                    [] if retained_component is None else sorted(retained_component)
                ),
                "archived_owner_id": owner_id if complete_handoff else None,
                "reservoir_id": reservoir_id,
                "fresh_component_owner_ids": sorted(fresh_owner_ids),
                "historical_state_transferred_to_components": False,
                "historical_state_divided_or_copied": False,
            },
        )
        if result.total_conserved_ledgers() != state.total_conserved_ledgers():
            raise RuntimeError("process handoff violated active+wake+reservoir conservation")
        if result.total_signed_system_ledgers() != state.total_signed_system_ledgers():
            raise RuntimeError("process handoff violated signed-system conservation")
    return result


def kernel_coordinate_for_owner(
    state: MultiFrontRuntimeState, owner_id: str, *,
    qualified_upper_bound_m: float | None = None,
) -> float:
    coordinate = state.process_regions[owner_id].cumulative_process_advance_m
    if qualified_upper_bound_m is None:
        engine_id = state.process_regions[owner_id].process_engine_id
        engine = state.process_engines[engine_id]
        if engine.checkpoint_payload_b64 is None:
            # Source-only fixtures retain the historical bound explicitly.
            qualified_upper_bound_m = 745.0e-6
        else:
            payload = engine.complete_checkpoint_payload()
            family = payload.get("engine_fields", {}).get("_state_kernel_family")
            states = getattr(family, "states", ())
            if not states:
                raise ValueError("owner process engine lacks a qualified kernel family")
            qualified_upper_bound_m = max(
                float(kernel_state.coordinates[2]) for kernel_state in states
            )
    if coordinate > float(qualified_upper_bound_m) + 1.0e-18:
        raise ValueError("owner_local_kernel_coordinate_outside_qualified_family_domain")
    return coordinate


__all__ = [
    "AcceptedIntervalResult", "BRANCHING_MODES", "BranchJunctionState",
    "CouplingEvidence", "FrontCandidateObservation", "FrontRuntimeState",
    "GlobalSchedulerState", "MultiFrontRuntimeState", "POLICY_BOUND_FRONT_LIMIT",
    "POLICY_BOUND_TRANSACTION_LIMIT", "PARENT_PROCESS_ZONE_UNRESOLVED",
    "ProcessEngineState", "ProcessRegionReservoir",
    "ProcessRegionState", "ResourcePolicy", "SCHEMA", "TopologyProposal",
    "TopologyTransactionRecord", "advance_accepted_interval", "bifurcate_binary",
    "canonical_hash", "canonical_json", "coalesce_front", "commit_selected_proposal",
    "extend_one_arm", "form_correlated_topology_proposals", "kernel_coordinate_for_owner",
    "recompute_process_region_connectivity", "reject_proposal", "retire_front",
    "select_global_topology_proposal",
]
