"""Explicit, reference-safe ownership of v11 moving-tip process state."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


OWNER_TYPES = frozenset({
    "single_tip_engine", "unresolved_branch_cluster",
    "resolved_tip_engine", "junction_reservoir",
})


@dataclass(frozen=True)
class ProcessStateOwner:
    owner_id: str
    owner_type: str
    process_engine_id: str | None = None
    cluster_id: str | None = None
    junction_reservoir_id: str | None = None

    def __post_init__(self) -> None:
        if self.owner_type not in OWNER_TYPES:
            raise ValueError(f"unsupported process-state owner type: {self.owner_type}")

    def to_dict(self) -> dict:
        return {
            "owner_id": self.owner_id, "owner_type": self.owner_type,
            "process_engine_id": self.process_engine_id,
            "cluster_id": self.cluster_id,
            "junction_reservoir_id": self.junction_reservoir_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "ProcessStateOwner":
        return cls(**{key: value.get(key) for key in (
            "owner_id", "owner_type", "process_engine_id", "cluster_id",
            "junction_reservoir_id",
        )})


@dataclass(frozen=True)
class ProcessStateOwnerRegistry:
    owner_by_tip: Mapping[str, str]
    owners: Mapping[str, ProcessStateOwner]

    def __post_init__(self) -> None:
        object.__setattr__(self, "owner_by_tip", dict(self.owner_by_tip))
        object.__setattr__(self, "owners", dict(self.owners))

    def owner(self, tip_id: str) -> ProcessStateOwner:
        return self.owners[self.owner_by_tip[tip_id]]

    def recursive_branch_eligible(self, tip_id: str) -> bool:
        return self.owner(tip_id).owner_type in {"single_tip_engine", "resolved_tip_engine"}

    def reference_count(self, owner_id: str) -> int:
        return sum(value == owner_id for value in self.owner_by_tip.values())

    def validate(
        self, active_tip_ids: Iterable[str], *, engine_ids: Iterable[str],
        cluster_ids: Iterable[str], reservoir_ids: Iterable[str],
    ) -> None:
        active = set(active_tip_ids)
        if set(self.owner_by_tip) != active:
            raise ValueError("every active front must have exactly one explicit process-state owner")
        engine_ids, cluster_ids, reservoir_ids = map(set, (engine_ids, cluster_ids, reservoir_ids))
        for tip_id in sorted(active):
            owner_id = self.owner_by_tip[tip_id]
            if owner_id not in self.owners:
                raise ValueError(f"active front {tip_id} references missing owner {owner_id}")
            owner = self.owners[owner_id]
            if owner.owner_type != "junction_reservoir" and owner.process_engine_id not in engine_ids:
                raise ValueError(f"active front {tip_id} references missing engine {owner.process_engine_id}")
            if owner.owner_type == "unresolved_branch_cluster" and owner.cluster_id not in cluster_ids:
                raise ValueError(f"unresolved owner {owner_id} has no cluster")
            if owner.owner_type in {"single_tip_engine", "resolved_tip_engine"} and self.reference_count(owner_id) != 1:
                raise ValueError("independent resolved process engines cannot be shared")
        for owner in self.owners.values():
            if owner.owner_type == "junction_reservoir" and owner.junction_reservoir_id not in reservoir_ids:
                raise ValueError(f"junction owner {owner.owner_id} has no reservoir")
        independent_engines = [
            owner.process_engine_id for owner in self.owners.values()
            if owner.owner_type in {"single_tip_engine", "resolved_tip_engine"}
        ]
        if len(independent_engines) != len(set(independent_engines)):
            raise ValueError("a mutable engine cannot back two independent tip owners")

    def to_dict(self) -> dict:
        return {
            "schema": "v11.process-state-owner-registry/1",
            "owner_by_tip": dict(self.owner_by_tip),
            "owners": {key: value.to_dict() for key, value in self.owners.items()},
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "ProcessStateOwnerRegistry":
        if value.get("schema") != "v11.process-state-owner-registry/1":
            raise ValueError("unsupported process-state owner registry")
        return cls(value["owner_by_tip"], {
            key: ProcessStateOwner.from_dict(item) for key, item in value["owners"].items()
        })

    def branch(self, parent_tip: str, cluster_id: str, child_tips: Iterable[str]) -> "ProcessStateOwnerRegistry":
        if not self.recursive_branch_eligible(parent_tip):
            raise ValueError("parent_process_zone_still_unresolved")
        mapping, owners = dict(self.owner_by_tip), dict(self.owners)
        prior_id = mapping.pop(parent_tip)
        if any(value == prior_id for value in mapping.values()):
            raise ValueError("independent owner retained an unexpected active reference")
        owners.pop(prior_id)
        owners[cluster_id] = ProcessStateOwner(
            cluster_id, "unresolved_branch_cluster", process_engine_id=cluster_id,
            cluster_id=cluster_id,
        )
        for tip in child_tips:
            mapping[tip] = cluster_id
        return ProcessStateOwnerRegistry(mapping, owners)

    def resolve(self, cluster_id: str, child_tips: Iterable[str], reservoir_id: str) -> "ProcessStateOwnerRegistry":
        mapping, owners = dict(self.owner_by_tip), dict(self.owners)
        children = tuple(child_tips)
        if self.owners[cluster_id].owner_type != "unresolved_branch_cluster":
            raise ValueError("only an unresolved cluster can resolve")
        if {tip for tip, owner in mapping.items() if owner == cluster_id} != set(children):
            raise ValueError("cluster resolution must transition all and only its active arms")
        owners.pop(cluster_id)
        owners[reservoir_id] = ProcessStateOwner(
            reservoir_id, "junction_reservoir", junction_reservoir_id=reservoir_id,
            cluster_id=cluster_id,
        )
        for tip in children:
            mapping[tip] = tip
            owners[tip] = ProcessStateOwner(tip, "resolved_tip_engine", process_engine_id=tip)
        return ProcessStateOwnerRegistry(mapping, owners)

    def retire_tip(self, tip_id: str) -> tuple["ProcessStateOwnerRegistry", str | None]:
        mapping, owners = dict(self.owner_by_tip), dict(self.owners)
        owner_id = mapping.pop(tip_id)
        removed = None
        if owner_id not in mapping.values():
            owner = owners[owner_id]
            if owner.owner_type != "junction_reservoir":
                owners.pop(owner_id)
                removed = owner.process_engine_id
        return ProcessStateOwnerRegistry(mapping, owners), removed


__all__ = ["OWNER_TYPES", "ProcessStateOwner", "ProcessStateOwnerRegistry"]
