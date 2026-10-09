"""Conservative unresolved branch-junction state for v11 branch birth."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .crack_network_v11 import CrackBranchState, CrackNetworkState


LEDGER_FIELDS = (
    "retained", "mobile", "escaped", "recovered", "stored_energy",
    "emission_work", "unconsumed_action",
)


def _ledger(values: Mapping[str, float]) -> dict[str, float]:
    result = {name: float(values.get(name, 0.0)) for name in LEDGER_FIELDS}
    if any(not math.isfinite(value) or value < 0.0 for value in result.values()):
        raise ValueError("branch-cluster ledgers must be finite and nonnegative")
    unknown = set(values).difference(LEDGER_FIELDS)
    if unknown:
        raise ValueError(f"unknown branch-cluster ledger fields: {sorted(unknown)}")
    return result


@dataclass(frozen=True)
class BranchClusterState:
    cluster_id: str
    junction_xy_m: tuple[float, float]
    parent_branch_id: str
    arm_branch_ids: tuple[str, str]
    shared_process_state: Mapping[str, Any]
    conserved_ledgers: Mapping[str, float]
    unresolved: bool = True

    def __post_init__(self) -> None:
        if len(set(self.arm_branch_ids)) != 2:
            raise ValueError("a v11 branch cluster requires two distinct arms")
        point = tuple(float(value) for value in self.junction_xy_m)
        if len(point) != 2 or not all(math.isfinite(value) for value in point):
            raise ValueError("branch junction must be a finite 2-D point")
        object.__setattr__(self, "junction_xy_m", point)
        object.__setattr__(self, "arm_branch_ids", tuple(sorted(self.arm_branch_ids)))
        object.__setattr__(self, "shared_process_state", json.loads(json.dumps(dict(self.shared_process_state), sort_keys=True, allow_nan=False)))
        object.__setattr__(self, "conserved_ledgers", _ledger(self.conserved_ledgers))

    def handoff(self, arm_ledgers: Sequence[Mapping[str, float]]) -> "BranchClusterState":
        """Resolve only when an externally supplied physical partition conserves every ledger."""
        if len(arm_ledgers) != 2:
            raise ValueError("independent-tip handoff requires two arm ledgers")
        normalized = tuple(_ledger(item) for item in arm_ledgers)
        for name, total in self.conserved_ledgers.items():
            if not math.isclose(math.fsum(item[name] for item in normalized), total, rel_tol=1e-12, abs_tol=1e-18):
                raise ValueError(f"nonconservative independent-tip handoff for {name}")
        return replace(
            self, unresolved=False,
            shared_process_state={**self.shared_process_state, "arm_ledgers": normalized},
        )


def _child_id(parent_id: str, candidate_id: str, event_index: int) -> str:
    token = f"{parent_id}|{candidate_id}|{int(event_index)}".encode("utf-8")
    return "b" + hashlib.sha256(token).hexdigest()[:15]


def create_unresolved_branch_cluster(
    network: CrackNetworkState,
    *,
    parent_branch_id: str,
    candidate_ids: tuple[str, str],
    event_index: int,
    shared_process_state: Mapping[str, Any],
    conserved_ledgers: Mapping[str, float],
) -> tuple[CrackNetworkState, BranchClusterState]:
    """Create two root-only arms while retaining one shared junction state."""
    if len(set(candidate_ids)) != 2:
        raise ValueError("branch birth requires two physical directions")
    parent = network.branch(parent_branch_id)
    if parent.status != "active":
        raise ValueError("branch parent must be active")
    ordered_candidates = tuple(sorted(candidate_ids))
    arm_ids = tuple(_child_id(parent_branch_id, item, event_index) for item in ordered_candidates)
    children = tuple(
        CrackBranchState(
            branch_id=arm_id, parent_branch_id=parent_branch_id,
            generation=parent.generation + 1, initiation_event=int(event_index),
            path=(parent.tip,), orientation_history_rad=(parent.current_orientation_rad,),
            local_state={"candidate_id": candidate_id, "cluster_unresolved": True},
        )
        for arm_id, candidate_id in zip(arm_ids, ordered_candidates)
    )
    retired_parent = replace(parent, status="terminated")
    branches = tuple(
        retired_parent if item.branch_id == parent_branch_id else item
        for item in network.branches
    ) + children
    born = CrackNetworkState(
        branches=branches, primary_branch_id=network.primary_branch_id,
        geometry_generation=network.geometry_generation,
        branching_enabled=True,
    )
    cluster_id = "j" + hashlib.sha256(
        f"{parent_branch_id}|{event_index}|{'|'.join(ordered_candidates)}".encode("utf-8")
    ).hexdigest()[:15]
    return born, BranchClusterState(
        cluster_id=cluster_id, junction_xy_m=parent.tip,
        parent_branch_id=parent_branch_id, arm_branch_ids=arm_ids,
        shared_process_state=shared_process_state,
        conserved_ledgers=conserved_ledgers,
    )


__all__ = ["BranchClusterState", "LEDGER_FIELDS", "create_unresolved_branch_cluster"]
