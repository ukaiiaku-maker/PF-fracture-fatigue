"""Atomic live-FEM topology trials for mechanistic v11 crack branching.

The transaction owns geometry and directional-event consumption.  Trial geometry
is built on an isolated snapshot, re-equilibrated at the accepted displacement
load, and committed only when its actual whole-body potential-energy release pays
the summed hazard-derived dissipation.  It introduces no fracture criterion.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, is_dataclass, replace
import math
import time
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from .crack_network_v11 import CrackBranchState, CrackNetworkState
from .crack_backend import SharpWakeBackend
from .causal_sharp_wake_v11 import apply_causal_segment
from .coalescence import segment_intersection_first
from .directional_competition_v11 import (
    CompetingActionProposal,
    DirectionalCompetitionState,
    accept_reservation,
    release_reservation,
    reserve_action,
)


MODEL_ID = "v11.monotonic_tip_only_live_fem_topology_transaction/1"


class FrozenMapping(dict):
    """Pickle-safe immutable mapping shared by topology siblings."""
    def _immutable(self, *args, **kwargs):
        raise TypeError("accepted topology mapping is immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return (_make_frozen_mapping, (dict(self),))


def _make_frozen_mapping(values: Mapping[str, Any]) -> FrozenMapping:
    frozen = dict.__new__(FrozenMapping)
    dict.update(frozen, values)
    return frozen


def _freeze(value: Any) -> Any:
    if isinstance(value, FrozenMapping):
        return value
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.flags.writeable:
            array = array.copy(); array.setflags(write=False)
        return array
    if isinstance(value, Mapping):
        return _make_frozen_mapping({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _freeze_array_container(value: Any, names: tuple[str, ...]) -> Any:
    changes = {}
    requires_clone = False
    for name in names:
        item = getattr(value, name, None)
        if isinstance(item, np.ndarray):
            changes[name] = _freeze(item)
            requires_clone |= item.flags.writeable
    if not changes:
        return value
    if not requires_clone:
        return value
    if is_dataclass(value):
        return replace(value, **changes)
    clone = copy.copy(value)
    for name, item in changes.items():
        setattr(clone, name, item)
    return clone


@dataclass(frozen=True)
class LiveFEMTopologyState:
    """Complete accepted state crossing a topology transaction boundary."""

    mesh: Any
    boundary: Any
    damage: np.ndarray
    displacement: np.ndarray
    ep_gp: np.ndarray
    rho_gp: np.ndarray
    elasticity_D: np.ndarray
    material: Any
    cohesive_network: Any
    crack_network: CrackNetworkState
    competition: DirectionalCompetitionState
    tip_process_state: Mapping[str, Any]
    junction_process_state: Mapping[str, Any]
    energy_ledgers: Mapping[str, float]
    rng_state: Any
    event_counters: Mapping[str, int]
    stored_energy_J_per_m: float

    def __post_init__(self) -> None:
        energy = float(self.stored_energy_J_per_m)
        if not math.isfinite(energy):
            raise ValueError("stored FEM energy must be finite")
        object.__setattr__(self, "stored_energy_J_per_m", energy)
        for name in ("damage", "displacement", "ep_gp", "rho_gp", "elasticity_D"):
            source = np.asarray(getattr(self, name), dtype=float)
            array = source if not source.flags.writeable else source.copy()
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        object.__setattr__(self, "mesh", _freeze_array_container(
            self.mesh, ("nodes", "elems", "area_e", "dNdx_e", "B_e", "element_damage_gp"),
        ))
        object.__setattr__(self, "boundary", _freeze_array_container(
            self.boundary, ("top_nodes", "bot_nodes", "notch_nodes"),
        ))
        object.__setattr__(self, "tip_process_state", _freeze(self.tip_process_state))
        object.__setattr__(self, "junction_process_state", _freeze(self.junction_process_state))
        object.__setattr__(self, "energy_ledgers", _freeze(self.energy_ledgers))
        object.__setattr__(self, "event_counters", _freeze(self.event_counters))
        object.__setattr__(self, "rng_state", _freeze(self.rng_state))

    def isolated_copy(self) -> "LiveFEMTopologyState":
        return LiveFEMTopologyState(
            mesh=self.mesh, boundary=self.boundary,
            damage=self.damage, displacement=self.displacement,
            ep_gp=self.ep_gp, rho_gp=self.rho_gp, elasticity_D=self.elasticity_D,
            material=self.material, cohesive_network=self.cohesive_network,
            crack_network=self.crack_network, competition=self.competition,
            tip_process_state=self.tip_process_state,
            junction_process_state=self.junction_process_state,
            energy_ledgers=self.energy_ledgers, rng_state=self.rng_state,
            event_counters=self.event_counters,
            stored_energy_J_per_m=self.stored_energy_J_per_m,
        )


@dataclass(frozen=True)
class TopologyArm:
    candidate_id: str
    branch_id: str
    start_xy_m: tuple[float, float]
    end_xy_m: tuple[float, float]
    event_reward_m: float
    hazard_dissipation_J_per_m: float

    def __post_init__(self) -> None:
        reward = float(self.event_reward_m)
        dissipation = float(self.hazard_dissipation_J_per_m)
        if reward <= 0.0 or not math.isfinite(reward):
            raise ValueError("each completed event must receive one positive physical reward")
        if dissipation < 0.0 or not math.isfinite(dissipation):
            raise ValueError("hazard-derived dissipation must be finite and nonnegative")
        if math.dist(self.start_xy_m, self.end_xy_m) <= 0.0:
            raise ValueError("topology arm must have positive geometric length")


@dataclass(frozen=True)
class TopologyTrialResult:
    accepted: bool
    state: LiveFEMTopologyState
    action_id: str
    energy_release_J_per_m: float
    hazard_dissipation_J_per_m: float
    energy_margin_J_per_m: float
    rejection_reason: str | None
    trial_copy_bytes: int = 0
    trial_copy_wall_time_s: float = 0.0


GeometryTrial = Callable[[LiveFEMTopologyState, tuple[TopologyArm, ...]], LiveFEMTopologyState]
EquilibrateTrial = Callable[[LiveFEMTopologyState], LiveFEMTopologyState]


def equilibrate_fixed_load_with_production_fem(
    state: LiveFEMTopologyState,
) -> LiveFEMTopologyState:
    """Use the installed production assembler at the accepted boundary opening."""
    from .hazard_energy_event_gate_v10230 import _equilibrate_fixed_opening

    displacement, _, energy = _equilibrate_fixed_opening(
        mesh=state.mesh, boundary=state.boundary,
        u_initial=state.displacement, ep_gp=state.ep_gp, rho_gp=state.rho_gp,
        damage=state.damage, D=state.elasticity_D, mat=state.material,
        cohesive_network=state.cohesive_network,
    )
    return replace(state, displacement=displacement, stored_energy_J_per_m=energy)


def apply_sharp_wake_trial_geometry(
    state: LiveFEMTopologyState,
    arms: tuple[TopologyArm, ...],
    *,
    kill_radius_m: float,
) -> LiveFEMTopologyState:
    """Insert every arm into one copied sharp-wake damage field."""
    backend = SharpWakeBackend()
    current = state
    for index, arm in enumerate(arms):
        result = backend.advance(
            mesh=current.mesh, boundary=current.boundary,
            damage=current.damage, displacement=current.displacement,
            p0=np.asarray(arm.start_xy_m), p1=np.asarray(arm.end_xy_m),
            direction=np.asarray(arm.end_xy_m) - np.asarray(arm.start_xy_m),
            front_id=index, kill_r=float(kill_radius_m),
        )
        if not result.inserted or not math.isclose(
            result.moved, arm.event_reward_m, rel_tol=1.0e-12, abs_tol=1.0e-18
        ):
            raise RuntimeError(f"sharp-wake topology insertion failed: {result.reason}")
        current = replace(
            current, mesh=result.mesh, boundary=result.boundary,
            damage=result.damage, displacement=result.displacement,
        )
    return current


def apply_causal_sharp_wake_trial_geometry(
    state: LiveFEMTopologyState,
    arms: tuple[TopologyArm, ...],
) -> LiveFEMTopologyState:
    """Insert v11 arms with causal P0 support and reject invisible trials."""
    current = state
    audits = []
    for arm in arms:
        current, audit = apply_causal_segment(
            current, np.asarray(arm.start_xy_m), np.asarray(arm.end_xy_m),
        )
        if not audit.mechanically_resolved:
            raise RuntimeError(
                "sharp_wake_trial_not_mechanically_resolved: "
                f"branch={arm.branch_id} candidate={arm.candidate_id}"
            )
        audits.append({
            "branch_id": arm.branch_id,
            "candidate_id": arm.candidate_id,
            **audit.__dict__,
        })
    junction = dict(current.junction_process_state)
    junction["latest_causal_support_trials"] = tuple(audits)
    junction["crack_representation"] = "sharp_wake_causal_v11"
    return replace(current, junction_process_state=junction)


def _replace_branch(network: CrackNetworkState, updated: CrackBranchState) -> CrackNetworkState:
    branches = tuple(updated if item.branch_id == updated.branch_id else item for item in network.branches)
    return replace(network, branches=branches, geometry_generation=network.geometry_generation + 1)


def extend_network_arm(network: CrackNetworkState, arm: TopologyArm) -> CrackNetworkState:
    """Append an accepted segment without splitting any process-zone ledger."""
    branch = network.branch(arm.branch_id)
    if branch.status != "active" or branch.tip != tuple(arm.start_xy_m):
        raise ValueError("arm must extend an active branch from its accepted tip")
    angle = math.atan2(
        arm.end_xy_m[1] - arm.start_xy_m[1],
        arm.end_xy_m[0] - arm.start_xy_m[0],
    )
    local = dict(branch.local_state)
    edges = list(local.get("committed_edges", ()))
    edges.append({
        "start_point_m": list(arm.start_xy_m),
        "end_point_m": list(arm.end_xy_m),
        "branch_id": arm.branch_id,
        "parent_or_junction_id": branch.parent_branch_id,
        "commit_event_id": int(network.geometry_generation + 1),
    })
    local["committed_edges"] = edges
    updated = replace(
        branch,
        path=branch.path + (tuple(arm.end_xy_m),),
        orientation_history_rad=(
            (angle,) if len(branch.path) == 1
            else branch.orientation_history_rad + (angle,)
        ),
        local_state=local,
    )
    return _replace_branch(network, updated)


def clip_arm_at_first_intersection(
    network: CrackNetworkState, arm: TopologyArm
) -> tuple[TopologyArm, str | None]:
    """Clip an incoming arm at its first exact hit without changing either path."""
    p0 = np.asarray(arm.start_xy_m, dtype=float)
    p1 = np.asarray(arm.end_xy_m, dtype=float)
    best: tuple[float, np.ndarray, str] | None = None
    for branch in network.branches:
        for index, (a, b) in enumerate(zip(branch.path, branch.path[1:])):
            if branch.branch_id == arm.branch_id and index == len(branch.path) - 2:
                continue
            hit = segment_intersection_first(p0, p1, np.asarray(a), np.asarray(b))
            if hit is not None and (best is None or hit[0] < best[0]):
                best = hit[0], hit[1], branch.branch_id
    if best is None:
        return arm, None
    _, point, target = best
    clipped = replace(
        arm, end_xy_m=(float(point[0]), float(point[1])),
        event_reward_m=float(np.linalg.norm(point - p0)),
    )
    return clipped, target


def mark_coalesced(network: CrackNetworkState, incoming_branch_id: str, target_branch_id: str) -> CrackNetworkState:
    """Deactivate only the incoming tip and preserve both committed paths."""
    if incoming_branch_id == target_branch_id:
        raise ValueError("a branch cannot coalesce into itself")
    network.branch(target_branch_id)
    incoming = network.branch(incoming_branch_id)
    if incoming.status != "active":
        raise ValueError("only an active incoming branch can coalesce")
    local = dict(incoming.local_state)
    local.update({"coalesced": True, "merge_target_branch_id": target_branch_id})
    return _replace_branch(network, replace(incoming, status="merged", local_state=local))


def execute_topology_trial(
    accepted: LiveFEMTopologyState,
    proposal: CompetingActionProposal,
    arms: Iterable[TopologyArm],
    *,
    apply_trial_geometry: GeometryTrial,
    equilibrate_fixed_load: EquilibrateTrial,
    relative_energy_tolerance: float = 1.0e-8,
    absolute_energy_tolerance_J_per_m: float = 1.0e-12,
    network_geometry_already_realized: bool = False,
) -> TopologyTrialResult:
    """Reserve, trial and atomically accept/release one one- or two-arm action."""
    trial_arms = tuple(sorted(arms, key=lambda item: item.candidate_id))
    if len(trial_arms) not in (1, 2):
        raise ValueError("v11 topology actions contain exactly one or two arms")
    if tuple(item.candidate_id for item in trial_arms) != proposal.member_candidate_ids:
        raise ValueError("trial arms do not match the proposal's physical candidates")
    rewards = tuple(item.event_reward_m for item in trial_arms)
    reserved = reserve_action(accepted.competition, proposal, event_rewards_m=rewards)
    reserved_state = replace(accepted, competition=reserved)
    copy_start = time.perf_counter()
    isolated = reserved_state.isolated_copy()
    copy_wall = time.perf_counter() - copy_start
    mechanics = ("damage", "displacement", "ep_gp", "rho_gp", "elasticity_D")
    copy_bytes = sum(
        int(getattr(isolated, name).nbytes)
        for name in mechanics if getattr(isolated, name) is not getattr(reserved_state, name)
    )
    trial = apply_trial_geometry(isolated, trial_arms)
    if network_geometry_already_realized:
        for arm in trial_arms:
            branch = trial.crack_network.branch(arm.branch_id)
            if branch.tip != tuple(arm.end_xy_m):
                raise RuntimeError(
                    "trial network does not contain the exact realized arm endpoint"
                )
    trial = equilibrate_fixed_load(trial)
    released = float(accepted.stored_energy_J_per_m - trial.stored_energy_J_per_m)
    dissipation = math.fsum(item.hazard_dissipation_J_per_m for item in trial_arms)
    tolerance = max(
        float(absolute_energy_tolerance_J_per_m),
        max(float(relative_energy_tolerance), 0.0)
        * max(abs(accepted.stored_energy_J_per_m), abs(trial.stored_energy_J_per_m), dissipation),
    )
    margin = released - dissipation
    if released + tolerance < dissipation:
        # The reservation existed only inside the isolated trial. Discarding that
        # snapshot is the exact release operation and leaves no transactional
        # history in accepted production state.
        release_reservation(reserved, proposal.action_id)
        return TopologyTrialResult(
            False, accepted, proposal.action_id, released, dissipation, margin,
            "insufficient_whole_topology_energy_release", copy_bytes, copy_wall,
        )
    committed_competition = accept_reservation(trial.competition, proposal.action_id)
    committed_network = trial.crack_network
    if not network_geometry_already_realized:
        for arm in trial_arms:
            committed_network = extend_network_arm(committed_network, arm)
    committed = replace(
        trial,
        competition=committed_competition,
        crack_network=committed_network,
        event_counters={
            **trial.event_counters,
            "topology_actions": int(trial.event_counters.get("topology_actions", 0)) + 1,
        },
        energy_ledgers={
            **trial.energy_ledgers,
            "topology_release_J_per_m": float(trial.energy_ledgers.get("topology_release_J_per_m", 0.0)) + released,
            "hazard_dissipation_J_per_m": float(trial.energy_ledgers.get("hazard_dissipation_J_per_m", 0.0)) + dissipation,
        },
    )
    return TopologyTrialResult(
        True, committed, proposal.action_id, released, dissipation, margin, None,
        copy_bytes, copy_wall,
    )


__all__ = [
    "LiveFEMTopologyState", "MODEL_ID", "TopologyArm", "TopologyTrialResult",
    "apply_sharp_wake_trial_geometry", "apply_causal_sharp_wake_trial_geometry",
    "clip_arm_at_first_intersection",
    "equilibrate_fixed_load_with_production_fem", "execute_topology_trial",
    "extend_network_arm", "mark_coalesced",
]
