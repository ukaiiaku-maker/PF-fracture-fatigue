"""Atomic live-FEM topology trials for mechanistic v11 crack branching.

The transaction owns geometry and directional-event consumption.  Trial geometry
is built on an isolated snapshot, re-equilibrated at the accepted displacement
load, and committed only when its actual whole-body potential-energy release pays
the summed hazard-derived dissipation.  It introduces no fracture criterion.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, is_dataclass, replace
import hashlib
import json
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
    sharp_wake_model_id: str = "sharp_wake_causal_v11"
    v12_support_state: Any = None
    checkpoint_generation: int = 0

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
        from .sharp_wake_backend_v12 import V11_MODEL_ID, V12_MODEL_ID, select_sharp_wake_model
        selected=select_sharp_wake_model(self.sharp_wake_model_id)
        if selected==V11_MODEL_ID and self.v12_support_state is not None:
            raise ValueError("V11 state may not own V12 support")
        if selected==V12_MODEL_ID and self.v12_support_state is None:
            raise ValueError("V12 state requires authoritative support ownership")
        object.__setattr__(self,"sharp_wake_model_id",selected)

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
            sharp_wake_model_id=self.sharp_wake_model_id,
            v12_support_state=self.v12_support_state,
            checkpoint_generation=self.checkpoint_generation,
        )


@dataclass(frozen=True)
class TopologyArm:
    candidate_id: str
    branch_id: str
    start_xy_m: tuple[float, float]
    end_xy_m: tuple[float, float]
    event_reward_m: float
    hazard_dissipation_J_per_m: float
    event_classification: str = "software_forced_geometry"
    candidate_direction_xy: tuple[float, float] | None = None
    first_intersection_xy_m: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        reward = float(self.event_reward_m)
        dissipation = float(self.hazard_dissipation_J_per_m)
        if reward <= 0.0 or not math.isfinite(reward):
            raise ValueError("each completed event must receive one positive physical reward")
        if dissipation < 0.0 or not math.isfinite(dissipation):
            raise ValueError("hazard-derived dissipation must be finite and nonnegative")
        if math.dist(self.start_xy_m, self.end_xy_m) <= 0.0:
            raise ValueError("topology arm must have positive geometric length")
        classification = str(self.event_classification)
        if classification not in {"software_forced_geometry", "physical_cleavage"}:
            raise ValueError("unknown topology-event classification")
        object.__setattr__(self, "event_classification", classification)
        if classification == "physical_cleavage":
            if dissipation <= 0.0:
                raise ValueError("a physical cleavage event requires nonzero hazard dissipation")
            if self.candidate_direction_xy is None or self.first_intersection_xy_m is None:
                raise ValueError("a physical cleavage event requires direction and first-intersection evidence")
            direction = np.asarray(self.candidate_direction_xy, dtype=float)
            segment = np.asarray(self.end_xy_m, dtype=float) - np.asarray(self.start_xy_m, dtype=float)
            if direction.shape != (2,) or not np.all(np.isfinite(direction)):
                raise ValueError("physical candidate direction must be a finite 2-vector")
            if np.linalg.norm(direction) <= 0.0:
                raise ValueError("physical candidate direction must be nonzero")
            cosine = float(np.dot(direction, segment) / (np.linalg.norm(direction) * np.linalg.norm(segment)))
            if cosine < 1.0 - 1.0e-10:
                raise ValueError("realized arm is not aligned with its selected physical candidate")
            if not np.allclose(self.end_xy_m, self.first_intersection_xy_m, rtol=0.0, atol=1.0e-15):
                raise ValueError("realized endpoint is not the recorded first intersection")


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
    """Equilibrate with the production assembler at the accepted opening.

    This deliberately calls the public FEM assembly and Dirichlet solver.  It
    does not depend on a test observer or replace energy with a synthetic value.
    """
    from .fem import assemble_mechanics, elastic_energy_densities, solve_dirichlet

    u0 = np.asarray(state.displacement, dtype=float).copy()
    top = np.asarray(state.boundary.top_nodes, dtype=int)
    bot = np.asarray(state.boundary.bot_nodes, dtype=int)
    top_opening = float(np.mean(u0[2 * top + 1])) if top.size else 0.0
    bottom_opening = float(np.mean(u0[2 * bot + 1])) if bot.size else 0.0
    Kmat, Rint, *_ = assemble_mechanics(
        state.mesh, u0, state.ep_gp, state.rho_gp, state.damage,
        state.elasticity_D, state.material,
        cohesive_network=state.cohesive_network,
    )
    displacement, reaction = solve_dirichlet(
        Kmat, Rint, u0, state.boundary, top_opening, bottom_opening,
    )
    _, residual, sigma_gp, *_ = assemble_mechanics(
        state.mesh, displacement, state.ep_gp, state.rho_gp, state.damage,
        state.elasticity_D, state.material,
        cohesive_network=state.cohesive_network,
    )
    density, _ = elastic_energy_densities(
        state.mesh, displacement, state.ep_gp, sigma_gp, state.elasticity_D,
    )
    energy = float(np.sum(density * state.mesh.area_e))
    ledger = dict(state.energy_ledgers)
    ledger.update({
        "latest_reaction_N_per_m": float(reaction),
        "latest_residual_l2_N_per_m": float(np.linalg.norm(residual)),
        "latest_fem_energy_J_per_m": energy,
    })
    return replace(
        state, displacement=displacement, stored_energy_J_per_m=energy,
        energy_ledgers=ledger,
    )


def complete_accepted_state_fingerprint(state: LiveFEMTopologyState) -> str:
    """Hash all accepted production ownership without object identities."""
    from .sharp_wake_backend_v12 import array_fingerprint

    def normalize(value):
        if isinstance(value, np.ndarray):
            return {"array_sha256": array_fingerprint(value)}
        if hasattr(value, "to_dict"):
            return normalize(value.to_dict())
        if is_dataclass(value):
            return normalize({name: getattr(value, name) for name in value.__dataclass_fields__})
        if isinstance(value, Mapping):
            return {str(key): normalize(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
        if isinstance(value, (tuple, list)):
            return [normalize(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, float) and not math.isfinite(value):
            return {"nonfinite": str(value)}
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return {"type": type(value).__qualname__, "state": normalize(getattr(value, "__dict__", str(value)))}

    payload = {
        "mesh_nodes": state.mesh.nodes,
        "mesh_elems": state.mesh.elems,
        "boundary": state.boundary,
        "damage": state.damage,
        "displacement": state.displacement,
        "ep_gp": state.ep_gp,
        "rho_gp": state.rho_gp,
        "elasticity_D": state.elasticity_D,
        "material": state.material,
        "cohesive_network": state.cohesive_network,
        "crack_network": state.crack_network,
        "competition": state.competition,
        "tip_process_state": state.tip_process_state,
        "junction_process_state": state.junction_process_state,
        "energy_ledgers": state.energy_ledgers,
        "rng_state": state.rng_state,
        "event_counters": state.event_counters,
        "stored_energy_J_per_m": state.stored_energy_J_per_m,
        "sharp_wake_model_id": state.sharp_wake_model_id,
        "v12_support_state": state.v12_support_state,
        "checkpoint_generation": state.checkpoint_generation,
    }
    encoded = json.dumps(normalize(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


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


def apply_mechanically_separating_v12_trial_geometry(
    state: LiveFEMTopologyState, arms: tuple[TopologyArm, ...], *,
    source_commit: str, configuration: Mapping[str, Any], transaction_identity: str,
    failure_injector: Callable[[str, LiveFEMTopologyState], None] | None = None,
) -> LiveFEMTopologyState:
    """Realize tentative graph first, then rebuild/certify complete V12 support."""
    from .causal_sharp_wake_v11 import element_damage
    from .mechanically_separating_sharp_wake_v12 import apply_mechanically_separating_graph
    from .sharp_wake_backend_v12 import V12_MODEL_ID, support_state_from_production
    network=state.crack_network
    for arm in arms: network=extend_network_arm(network,arm)
    if failure_injector is not None: failure_injector("graph_edit",replace(state,crack_network=network))
    previous=None
    if state.v12_support_state is not None:
        previous=state.junction_process_state.get("v12_support_record")
    trial,audit=apply_mechanically_separating_graph(state,network,previous_support=previous)
    if failure_injector is not None: failure_injector("support_generation",trial)
    if not audit.certified or audit.mechanically_new_element_count<=0:
        raise RuntimeError("V12 tentative event did not create certified mechanical novelty")
    if failure_injector is not None: failure_injector("support_certification",trial)
    prior=state.v12_support_state.transaction_identity if state.v12_support_state is not None else None
    ownership=support_state_from_production(
        mesh=trial.mesh,crack_network=network,selected_support_elements=audit.selected_element_ids,
        damage_gp=element_damage(trial.mesh,trial.damage),certification_fingerprint=audit.certificate_fingerprint,
        transaction_identity=transaction_identity,previous_accepted_transaction=prior,
        source_commit=source_commit,configuration=configuration,
        checkpoint_generation=state.checkpoint_generation)
    return replace(trial,sharp_wake_model_id=V12_MODEL_ID,v12_support_state=ownership)


def initialize_mechanically_separating_v12(
    state: LiveFEMTopologyState, *, source_commit: str,
    configuration: Mapping[str, Any], transaction_identity: str="v12-initial",
) -> LiveFEMTopologyState:
    """Select V12 only together with certified support for the accepted graph."""
    from .causal_sharp_wake_v11 import element_damage
    from .mechanically_separating_sharp_wake_v12 import apply_mechanically_separating_graph
    from .sharp_wake_backend_v12 import V12_MODEL_ID, support_state_from_production
    trial,audit=apply_mechanically_separating_graph(state,state.crack_network)
    if not audit.certified: raise RuntimeError("initial V12 support is not certified")
    ownership=support_state_from_production(
        mesh=trial.mesh,crack_network=trial.crack_network,
        selected_support_elements=audit.selected_element_ids,
        damage_gp=element_damage(trial.mesh,trial.damage),
        certification_fingerprint=audit.certificate_fingerprint,
        transaction_identity=transaction_identity,previous_accepted_transaction=None,
        source_commit=source_commit,configuration=configuration,
        checkpoint_generation=state.checkpoint_generation)
    return replace(trial,sharp_wake_model_id=V12_MODEL_ID,v12_support_state=ownership)


def remesh_mechanically_separating_v12(
    state: LiveFEMTopologyState, *, mesh, boundary, tentative_network=None,
    transferred_fields: Mapping[str, Any], source_commit: str,
    configuration: Mapping[str, Any], transaction_identity: str,
    failure_injector: Callable[[str, LiveFEMTopologyState], None] | None=None,
) -> LiveFEMTopologyState:
    """Rebuild V12 ownership on a new mesh; never transfer element IDs."""
    required=("damage","displacement","ep_gp","rho_gp","tip_process_state","source_state")
    if any(name not in transferred_fields for name in required):
        raise ValueError("V12 remesh requires every owned structural/history field")
    junction={key:value for key,value in state.junction_process_state.items()
              if key not in ("v12_support_record","v12_graph_support_audit",
                             "v12_accepted_mechanical_fingerprint","v12_trial_mechanical_fingerprint")}
    counters = dict(state.event_counters)
    counters["mesh_generation"] = int(counters.get("mesh_generation", 0)) + 1
    counters["refinement_operation_index"] = int(counters.get("refinement_operation_index", 0)) + 1
    base=replace(state,mesh=mesh,boundary=boundary,
        crack_network=state.crack_network if tentative_network is None else tentative_network,
        damage=np.asarray(transferred_fields["damage"]),
        displacement=np.asarray(transferred_fields["displacement"]),
        ep_gp=np.asarray(transferred_fields["ep_gp"]),rho_gp=np.asarray(transferred_fields["rho_gp"]),
        tip_process_state=transferred_fields["tip_process_state"],
        junction_process_state={**junction, "source_state": transferred_fields["source_state"]},
        event_counters=counters, sharp_wake_model_id="sharp_wake_causal_v11",v12_support_state=None)
    if failure_injector is not None: failure_injector("remesh",base)
    if failure_injector is not None: failure_injector("field_projection",base)
    rebuilt = initialize_mechanically_separating_v12(
        base,source_commit=source_commit,configuration=configuration,
        transaction_identity=transaction_identity)
    if failure_injector is not None: failure_injector("support_rebuild",rebuilt)
    return rebuilt


def apply_v12_production_trial_geometry(
    state: LiveFEMTopologyState,
    arms: tuple[TopologyArm, ...],
    *,
    source_commit: str,
    configuration: Mapping[str, Any],
    transaction_identity: str,
    failure_injector: Callable[[str, LiveFEMTopologyState], None] | None = None,
) -> LiveFEMTopologyState:
    """Perform graph edit, conforming remesh, physical field transfer and support rebuild."""
    from .adaptive_multitip_mesh_v11 import refine_accepted_state

    network = state.crack_network
    for arm in arms:
        network = extend_network_arm(network, arm)
    graph_state = replace(state, crack_network=network)
    if failure_injector is not None:
        failure_injector("graph_edit", graph_state)

    graph_segments = tuple(
        (a, b) for branch in network.branches for a, b in zip(branch.path, branch.path[1:])
    )
    refined = graph_state
    for level in range(3):
        centroids = np.asarray(refined.mesh.nodes)[np.asarray(refined.mesh.elems)].mean(axis=1)
        marked: set[int] = set()
        for start, end in graph_segments:
            a = np.asarray(start, dtype=float)
            b = np.asarray(end, dtype=float)
            delta = b - a
            length2 = float(delta @ delta)
            t = np.clip(((centroids - a) @ delta) / max(length2, 1.0e-300), 0.0, 1.0)
            distance = np.linalg.norm(centroids - (a + t[:, None] * delta), axis=1)
            local_h = np.sqrt(np.maximum(np.asarray(refined.mesh.area_e), 1.0e-300))
            marked.update(np.flatnonzero(distance <= 2.0 * local_h).tolist())
        if not marked:
            raise RuntimeError("production V12 remesh found no physical event support")
        refined, _ = refine_accepted_state(
            refined,
            marked_parent_elements=tuple(sorted(marked)),
            active_tip_ids=network.active_tip_ids,
            generation=int(state.event_counters.get("mesh_generation", 0)) + 1,
            operation_index=int(state.event_counters.get("refinement_operation_index", 0)) + level + 1,
        )
    counters = dict(refined.event_counters)
    counters.update({
        "mesh_generation": int(state.event_counters.get("mesh_generation", 0)) + 1,
        "refinement_operation_index": int(state.event_counters.get("refinement_operation_index", 0)) + 1,
    })
    refined = replace(refined, event_counters=counters)
    return remesh_mechanically_separating_v12(
        state,
        mesh=refined.mesh,
        boundary=refined.boundary,
        tentative_network=network,
        transferred_fields={
            "damage": refined.damage,
            "displacement": refined.displacement,
            "ep_gp": refined.ep_gp,
            "rho_gp": refined.rho_gp,
            "tip_process_state": state.tip_process_state,
            "source_state": state.junction_process_state.get("source_state", {}),
        },
        source_commit=source_commit,
        configuration=configuration,
        transaction_identity=transaction_identity,
        failure_injector=failure_injector,
    )


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
    failure_injector: Callable[[str, LiveFEMTopologyState], None] | None = None,
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
    if failure_injector is not None: failure_injector("accepted_snapshot",isolated)
    copy_wall = time.perf_counter() - copy_start
    mechanics = ("damage", "displacement", "ep_gp", "rho_gp", "elasticity_D")
    copy_bytes = sum(
        int(getattr(isolated, name).nbytes)
        for name in mechanics if getattr(isolated, name) is not getattr(reserved_state, name)
    )
    trial = apply_trial_geometry(isolated, trial_arms)
    if failure_injector is not None: failure_injector("field_transfer",trial)
    if network_geometry_already_realized:
        for arm in trial_arms:
            branch = trial.crack_network.branch(arm.branch_id)
            if branch.tip != tuple(arm.end_xy_m):
                raise RuntimeError(
                    "trial network does not contain the exact realized arm endpoint"
                )
    trial = equilibrate_fixed_load(trial)
    if failure_injector is not None: failure_injector("equilibrium",trial)
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
    if failure_injector is not None: failure_injector("energy_gate",trial)
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
    if failure_injector is not None:
        failure_injector("process_state_update",committed)
        failure_injector("topology_verification",committed)
        failure_injector("late_event_veto",committed)
    return TopologyTrialResult(
        True, committed, proposal.action_id, released, dissipation, margin, None,
        copy_bytes, copy_wall,
    )


__all__ = [
    "LiveFEMTopologyState", "MODEL_ID", "TopologyArm", "TopologyTrialResult",
    "apply_sharp_wake_trial_geometry", "apply_causal_sharp_wake_trial_geometry", "apply_mechanically_separating_v12_trial_geometry", "apply_v12_production_trial_geometry", "initialize_mechanically_separating_v12", "remesh_mechanically_separating_v12",
    "clip_arm_at_first_intersection",
    "complete_accepted_state_fingerprint", "equilibrate_fixed_load_with_production_fem", "execute_topology_trial",
    "extend_network_arm", "mark_coalesced",
]
