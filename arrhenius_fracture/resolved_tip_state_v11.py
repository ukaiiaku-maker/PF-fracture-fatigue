"""Physical ownership transition from one shared branch cluster to independent tips."""
from __future__ import annotations

from dataclasses import dataclass, replace
import copy
import hashlib
from typing import Any, Callable, Mapping, Sequence

from .branch_cluster_v11 import BranchClusterState
from .crack_network_v11 import CrackNetworkState
from .directional_competition_v11 import (
    CleavageCandidate, DirectionalCompetitionState,
)


@dataclass(frozen=True)
class JunctionWakeReservoir:
    """Historical material state left behind at a resolved branch junction."""

    reservoir_id: str
    cluster_id: str
    junction_xy_m: tuple[float, float]
    historical_process_state: Mapping[str, Any]
    historical_ledgers: Mapping[str, float]
    arm_branch_ids: tuple[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "historical_process_state", copy.deepcopy(dict(self.historical_process_state)))
        object.__setattr__(self, "historical_ledgers", {
            key: float(value) for key, value in self.historical_ledgers.items()
        })


@dataclass(frozen=True)
class IndependentTipState:
    """Forward-local state born by ordinary fresh-tip renewal, never by splitting."""

    branch_id: str
    competition: DirectionalCompetitionState
    process_state: Mapping[str, Any]
    source_state: Mapping[str, Any]
    rng_identity: str
    initialization_policy: str = "fresh_moving_tip_renewal_no_historical_partition"

    def __post_init__(self) -> None:
        if self.initialization_policy != "fresh_moving_tip_renewal_no_historical_partition":
            raise ValueError("resolved tips require the physical fresh-tip initialization policy")
        object.__setattr__(self, "process_state", copy.deepcopy(dict(self.process_state)))
        object.__setattr__(self, "source_state", copy.deepcopy(dict(self.source_state)))


@dataclass(frozen=True)
class ClusterResolution:
    network: CrackNetworkState
    cluster: BranchClusterState
    reservoir: JunctionWakeReservoir
    tips: Mapping[str, IndependentTipState]


FreshTipFactory = Callable[[str], tuple[Mapping[str, Any], Mapping[str, Any]]]


def _tip_seed(global_seed: int, cluster_id: str, branch_id: str) -> int:
    digest = hashlib.sha256(f"{int(global_seed)}|{cluster_id}|{branch_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def resolve_unresolved_cluster(
    network: CrackNetworkState,
    cluster: BranchClusterState,
    *,
    candidates: Sequence[CleavageCandidate],
    global_hazard_seed: int,
    fresh_tip_factory: FreshTipFactory,
) -> ClusterResolution:
    """Archive all history at the junction and establish zero-history child tips.

    No value in ``cluster.conserved_ledgers`` or ``shared_process_state`` is
    copied into either child.  Those complete mappings move intact into the
    junction reservoir.  The callback must construct the same fresh local state
    used when ordinary moving-tip renewal establishes a new forward cell.
    """
    if not cluster.unresolved:
        raise ValueError("branch cluster is already resolved")
    active = set(network.active_tip_ids)
    if not set(cluster.arm_branch_ids).issubset(active):
        raise ValueError("resolution requires both cluster arms to remain active")
    reservoir = JunctionWakeReservoir(
        reservoir_id=f"reservoir:{cluster.cluster_id}", cluster_id=cluster.cluster_id,
        junction_xy_m=cluster.junction_xy_m,
        historical_process_state=cluster.shared_process_state,
        historical_ledgers=cluster.conserved_ledgers,
        arm_branch_ids=cluster.arm_branch_ids,
    )
    branches = []
    tips = {}
    for branch in network.branches:
        if branch.branch_id not in cluster.arm_branch_ids:
            branches.append(branch)
            continue
        local = dict(branch.local_state)
        local.update({
            "cluster_unresolved": False, "resolved_from_cluster": cluster.cluster_id,
            "junction_reservoir_id": reservoir.reservoir_id,
        })
        branches.append(replace(branch, local_state=local))
        process_state, source_state = fresh_tip_factory(branch.branch_id)
        if process_state.get("historical_state_imported") is not False:
            raise ValueError("fresh child process state must explicitly reject historical import")
        if source_state.get("historical_state_imported") is not False:
            raise ValueError("fresh child source state must explicitly reject historical import")
        seed = _tip_seed(global_hazard_seed, cluster.cluster_id, branch.branch_id)
        tips[branch.branch_id] = IndependentTipState(
            branch_id=branch.branch_id,
            competition=DirectionalCompetitionState.initialize(candidates, global_hazard_seed=seed),
            process_state=process_state, source_state=source_state,
            rng_identity=f"sha256:{seed:016x}",
        )
    resolved_network = replace(network, branches=tuple(branches))
    resolved_cluster = replace(
        cluster, unresolved=False,
        shared_process_state={
            "ownership": "junction_wake_reservoir",
            "reservoir_id": reservoir.reservoir_id,
        },
    )
    return ClusterResolution(
        network=resolved_network, cluster=resolved_cluster,
        reservoir=reservoir, tips=tips,
    )


__all__ = [
    "ClusterResolution", "IndependentTipState", "JunctionWakeReservoir",
    "resolve_unresolved_cluster",
]
