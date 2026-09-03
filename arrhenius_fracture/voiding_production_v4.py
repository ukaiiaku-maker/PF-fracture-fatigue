"""Executable one-void trajectory owned by the V12 production FEM state."""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .crack_network_v11 import CrackBranchState, CrackNetworkState, ROOT_BRANCH_ID
from .directional_competition_v11 import commit_directional_interval, preview_directional_interval
from .explicit_cavity_v4 import build_explicit_hole_mesh, fill_explicit_hole_mesh
from .mesh import rebuild_tri_mesh
from .fem import assemble_mechanics, plane_strain_D
from .sharp_front import make_emergent_config
from .sharp_wake_backend_v12 import V12_MODEL_ID
from .topology_transaction_v11 import (
    LiveFEMTopologyState, TopologyArm, apply_v12_production_trial_geometry,
    complete_accepted_state_fingerprint, equilibrate_fixed_load_with_production_fem,
    execute_topology_trial, initialize_mechanically_separating_v12,
    remesh_mechanically_separating_v12,
)
from .v12_production_driver import _competition
from .voiding_v4 import (
    Cavity2D, HazardClock, ProductionVoidState, VoidPhase, VoidSite, VoidingConfig,
    advance_site, arrhenius_rates, create_subgrid_cavity, grow_cavity_2d,
    promote_cavity, replace_cavity,
)

SCHEMA = "v12.production-one-void-trajectory/4"


def _head():
    return subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=Path(__file__).resolve().parents[1], text=True).strip()


def _geometry(radius_m=5.0e-5):
    hole = build_explicit_hole_mesh(1.0e-3, 1.0e-3, (7.0e-4, 0.0), radius_m, 5.0e-5, 32, radial_layers_override=12)
    return hole, fill_explicit_hole_mesh(hole)


def build_production_void_state(*, enabled=True, stochastic=False, seed=3621):
    hole, filled = _geometry()
    mesh = filled.mesh
    ray = 16
    start = tuple(map(float, mesh.nodes[12 * 32 + ray]))
    tip = tuple(map(float, mesh.nodes[3 * 32 + ray]))
    cfg = make_emergent_config()
    u = np.zeros(mesh.ndof)
    u[2 * np.asarray(filled.boundary.top_nodes) + 1] = 2.0e-7
    u[2 * np.asarray(filled.boundary.bot_nodes) + 1] = -2.0e-7
    void_state = None
    if enabled:
        rng = np.random.default_rng(seed)
        thresholds = tuple(float(rng.exponential()) for _ in range(3)) if stochastic else (0.25, 0.25, 0.35)
        site = VoidSite(
            "site-1", (7.0e-4, 0.0), VoidPhase.AVAILABLE_SITE, 0, 2, 0.8,
            HazardClock(0.0, thresholds[0]), HazardClock(0.0, thresholds[1]), HazardClock(0.0, thresholds[2]),
        )
        void_state = ProductionVoidState((site,), rng_state=rng.bit_generator.state)
    state = LiveFEMTopologyState(
        mesh, filled.boundary, np.zeros(mesh.nn), u,
        np.vstack((np.full(mesh.ne, 1.0e-5), np.full(mesh.ne, -0.5e-5), np.full(mesh.ne, 0.25e-5))),
        np.linspace(1.0e12, 1.2e12, mesh.ne), plane_strain_D(cfg.material), cfg.material, None,
        CrackNetworkState.one_tip((start, tip)), _competition(seed),
        {"retained": 4.0, "mobile": 1.0}, {"source_state": {"density": 3.0, "clock": 0.125}},
        {"emission_work": 1.0}, np.random.default_rng(seed).bit_generator.state,
        {"accepted_steps": 0, "topology_actions": 0, "mesh_generation": 0}, 0.0,
        void_state=void_state,
    )
    state = initialize_mechanically_separating_v12(
        state, source_commit=_head(), configuration={"voiding_enabled": enabled, "model": V12_MODEL_ID},
    )
    return equilibrate_fixed_load_with_production_fem(state), hole


def observables(state, operation):
    cavity = None if state.void_state is None or not state.void_state.cavities else state.void_state.cavities[0]
    site = None if state.void_state is None else state.void_state.sites[0]
    tri = np.asarray(state.mesh.nodes)[np.asarray(state.mesh.elems)]
    side = np.linalg.norm(tri[:, [1, 2, 0]] - tri[:, [0, 1, 2]], axis=2)
    avec, bvec = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    area = np.abs(avec[:, 0] * bvec[:, 1] - avec[:, 1] * bvec[:, 0]) / 2.0
    quality = 4.0 * np.sqrt(3.0) * area / np.maximum(np.sum(side**2, axis=1), 1.0e-300)
    reaction = float(state.energy_ledgers.get("latest_reaction_N_per_m", 0.0))
    return {
        "operation": operation, "fingerprint": complete_accepted_state_fingerprint(state),
        "mesh_nodes": int(state.mesh.nn), "mesh_elements": int(state.mesh.ne),
        "graph_length_m": float(state.crack_network.total_physical_crack_length_m),
        "reaction_N_per_m": reaction,
        "compliance_m2_per_N": 4.0e-7 / max(abs(reaction), 1.0e-300),
        "energy_J_per_m": float(state.stored_energy_J_per_m),
        "residual_N_per_m": float(state.energy_ledgers.get("latest_residual_l2_N_per_m", 0.0)),
        "void_phase": None if cavity is None else cavity.phase.value,
        "site_phase": None if site is None else site.phase.value,
        "cavity_radius_m": None if cavity is None else cavity.radius_m,
        "cavity_area_m2": None if cavity is None else cavity.area_m2,
        "inventory_area_m2": None if cavity is None else cavity.inventory_area_m2,
        "void_event_history": [] if state.void_state is None else list(state.void_state.event_history),
        "event_counters": dict(state.event_counters),
        "mesh_minimum_quality": float(np.min(quality)),
        "mesh_maximum_aspect_ratio": float(np.max(side, axis=1).max() / max(np.min(side), 1.0e-300)),
        "field_transfer_audit": state.junction_process_state.get("latest_void_remesh_audit"),
    }


def local_site_tensor(state):
    _, _, sigma, *_ = assemble_mechanics(
        state.mesh, state.displacement, state.ep_gp, state.rho_gp, state.damage,
        state.elasticity_D, state.material, cohesive_network=state.cohesive_network,
    )
    center = np.asarray(state.void_state.sites[0].center_m)
    centroids = state.mesh.nodes[state.mesh.elems].mean(axis=1)
    element = int(np.argmin(np.linalg.norm(centroids - center, axis=1)))
    return np.array([[sigma[0, element], sigma[2, element]], [sigma[2, element], sigma[1, element]]])


def cavity_boundary_tensor(state):
    """Average production stresses in elements incident to the explicit cavity boundary."""
    _, _, sigma, *_ = assemble_mechanics(
        state.mesh, state.displacement, state.ep_gp, state.rho_gp, state.damage,
        state.elasticity_D, state.material, cohesive_network=state.cohesive_network,
    )
    cavity = state.void_state.cavities[0]
    center = np.asarray(cavity.center_m)
    radii = np.linalg.norm(np.asarray(state.mesh.nodes) - center, axis=1)
    boundary_nodes = np.flatnonzero(radii <= np.min(radii) * (1.0 + 1.0e-7))
    elements = np.flatnonzero(np.any(np.isin(state.mesh.elems, boundary_nodes), axis=1))
    mean = np.mean(sigma[:, elements], axis=1)
    return np.array([[mean[0], mean[2]], [mean[2], mean[1]]]), tuple(map(int, elements))


def _project_fields(state, mesh):
    old_nodes = np.asarray(state.mesh.nodes); new_nodes = np.asarray(mesh.nodes)
    _, node_parent = cKDTree(old_nodes).query(new_nodes)
    old_centers = old_nodes[state.mesh.elems].mean(axis=1)
    new_centers = new_nodes[mesh.elems].mean(axis=1)
    _, element_parent = cKDTree(old_centers).query(new_centers)
    return {
        "damage": np.asarray(state.damage)[node_parent],
        "displacement": np.asarray(state.displacement).reshape(-1, 2)[node_parent].reshape(-1),
        "ep_gp": np.asarray(state.ep_gp)[:, element_parent],
        "rho_gp": np.asarray(state.rho_gp)[element_parent],
        "tip_process_state": state.tip_process_state,
        "source_state": state.junction_process_state.get("source_state", {}),
    }


def _grow_hole_boundary(hole, radius_m):
    nodes = np.asarray(hole.mesh.nodes).copy()
    count = len(hole.prescribed_polygon_nodes)
    theta = 2.0 * np.pi * np.arange(count) / count
    polygon_radius = float(radius_m) / math.cos(math.pi / count)
    center = np.asarray(hole.center_m)
    nodes[:count] = center + polygon_radius * np.c_[np.cos(theta), np.sin(theta)]
    mesh = rebuild_tri_mesh(nodes, np.asarray(hole.mesh.elems), tip_centers=np.asarray(hole.center_m))
    return replace(hole, mesh=mesh, radius_m=float(radius_m))


def remesh_cavity(state, hole, void_state, identity, operation_log=None, failure_stage=None):
    operations = operation_log if operation_log is not None else []
    trial = replace(state, void_state=void_state)
    fields = _project_fields(trial, hole.mesh)
    transfer_audit = {
        "source_displacement_l2_m": float(np.linalg.norm(state.displacement)),
        "projected_displacement_l2_m": float(np.linalg.norm(fields["displacement"])),
        "source_plastic_history_l2": float(np.linalg.norm(state.ep_gp)),
        "projected_plastic_history_l2": float(np.linalg.norm(fields["ep_gp"])),
        "source_density_l2_m2": float(np.linalg.norm(state.rho_gp)),
        "projected_density_l2_m2": float(np.linalg.norm(fields["rho_gp"])),
        "projected_fields_nonzero": bool(
            np.linalg.norm(fields["displacement"]) > 0.0 and np.linalg.norm(fields["ep_gp"]) > 0.0
            and np.linalg.norm(fields["rho_gp"]) > 0.0
        ),
    }
    def inject(stage, current):
        operations.append(stage)
        if stage == failure_stage: raise RuntimeError("injected:" + stage)
    rebuilt = remesh_mechanically_separating_v12(
        trial, mesh=hole.mesh, boundary=hole.boundary, transferred_fields=fields,
        source_commit=_head(), configuration={"voiding_enabled": True},
        transaction_identity=identity, failure_injector=inject,
    )
    junction = dict(rebuilt.junction_process_state)
    junction["latest_void_remesh_audit"] = transfer_audit
    rebuilt = replace(rebuilt, junction_process_state=junction)
    solved = equilibrate_fixed_load_with_production_fem(rebuilt)
    operations.append("equilibrium")
    if failure_stage == "equilibrium": raise RuntimeError("injected:equilibrium")
    return solved


def _complete_next_clock(state, start_time=1.0):
    hazards = list(state.competition.hazard_states)
    for index, hazard in enumerate(hazards):
        if not hazard.pending_events:
            hazards[index] = commit_directional_interval(
                hazard, preview_directional_interval(hazard, lambda_per_s=1.0,
                                                     start_time_s=start_time, duration_s=1.0),
            )
            break
    return replace(state, competition=replace(state.competition, hazard_states=tuple(hazards)))


def ligament_transaction(state, *, failure_stage=None, operation_log=None):
    cavity = state.void_state.cavities[0]
    start = state.crack_network.branch(ROOT_BRANCH_ID).tip
    target = np.asarray((cavity.center_m[0] - cavity.radius_m, cavity.center_m[1]))
    node = int(np.argmin(np.linalg.norm(np.asarray(state.mesh.nodes) - target, axis=1)))
    end = tuple(map(float, state.mesh.nodes[node]))
    proposal = next(item for item in __import__(
        "arrhenius_fracture.directional_competition_v11", fromlist=["construct_action_proposals"]
    ).construct_action_proposals(state.competition.hazard_states, correlation_interval_s=0.0) if item.action_type == "one_arm")
    arm = TopologyArm(proposal.member_candidate_ids[0], ROOT_BRANCH_ID, start, end, math.dist(start, end), 0.0)
    operations = operation_log if operation_log is not None else []
    def inject(stage, current):
        operations.append(stage)
        if stage == failure_stage: raise RuntimeError("injected:" + stage)
    def geometry(trial, arms):
        return apply_v12_production_trial_geometry(
            trial, arms, source_commit=_head(), configuration={"event": "CRACK_TO_VOID_LIGAMENT"},
            transaction_identity="ligament", failure_injector=inject,
        )
    result = execute_topology_trial(
        state, proposal, (arm,), apply_trial_geometry=geometry,
        equilibrate_fixed_load=equilibrate_fixed_load_with_production_fem,
        network_geometry_already_realized=True, failure_injector=inject,
    )
    if not result.accepted: raise RuntimeError("ligament event rejected")
    connected = replace(cavity, phase=VoidPhase.CONNECTED_VOID, lineage=cavity.lineage + ("CRACK_TO_VOID_LIGAMENT",))
    void_state = replace(
        replace_cavity(result.state.void_state, connected),
        event_history=result.state.void_state.event_history + ({"event": "CRACK_TO_VOID_LIGAMENT"},),
    )
    return replace(result.state, void_state=void_state), result


def downstream_front_transaction(state, *, continuation=False):
    state = _complete_next_clock(state, start_time=2.0 if continuation else 1.0)
    proposal = next(item for item in __import__(
        "arrhenius_fracture.directional_competition_v11", fromlist=["construct_action_proposals"]
    ).construct_action_proposals(state.competition.hazard_states, correlation_interval_s=0.0) if item.action_type == "one_arm")
    child_id = "void-front-1"
    center = np.asarray(state.void_state.cavities[0].center_m)
    nodes = np.asarray(state.mesh.nodes)
    right_boundary = int(np.argmin(np.linalg.norm(nodes - (center + [state.void_state.cavities[0].radius_m, 0.0]), axis=1)))
    if continuation:
        start = state.crack_network.branch(child_id).tip
        candidates = np.flatnonzero(nodes[:, 0] > start[0] + 1.0e-8)
        end_node = int(candidates[np.argmin(np.linalg.norm(nodes[candidates] - (np.asarray(start) + [7.5e-5, 0.0]), axis=1))])
        base_network = state.crack_network
    else:
        start = tuple(map(float, nodes[right_boundary]))
        child = CrackBranchState(child_id, ROOT_BRANCH_ID, 1,
                                 int(state.event_counters.get("topology_actions", 0)) + 1,
                                 (start,), (0.0,), local_state={"source": "direct_cavity_boundary_tensor"})
        base_network = replace(state.crack_network, branches=state.crack_network.branches + (child,),
                               geometry_generation=state.crack_network.geometry_generation + 1,
                               branching_enabled=True)
        candidates = np.flatnonzero(nodes[:, 0] > start[0] + 1.0e-8)
        end_node = int(candidates[np.argmin(np.linalg.norm(nodes[candidates] - (np.asarray(start) + [1.5e-4, 0.0]), axis=1))])
    end = tuple(map(float, nodes[end_node]))
    arm = TopologyArm(proposal.member_candidate_ids[0], child_id, start, end, math.dist(start, end), 0.0)
    operations = []
    def inject(stage, current): operations.append(stage)
    def geometry(trial, arms):
        return apply_v12_production_trial_geometry(
            replace(trial, crack_network=base_network), arms,
            source_commit=_head(), configuration={"event": "DOWNSTREAM_FRONT" if not continuation else "CONTINUED_FRONT"},
            transaction_identity="downstream-continued" if continuation else "downstream-first-passage",
            failure_injector=inject, refinement_levels=1,
        )
    result = execute_topology_trial(
        state, proposal, (arm,), apply_trial_geometry=geometry,
        equilibrate_fixed_load=equilibrate_fixed_load_with_production_fem,
        network_geometry_already_realized=True, failure_injector=inject,
    )
    if not result.accepted: raise RuntimeError("downstream front event rejected")
    cavity = result.state.void_state.cavities[0]
    phase = VoidPhase.DOWNSTREAM_FRONT_ACTIVE
    updated = replace(cavity, phase=phase, lineage=cavity.lineage + (("CONTINUED_EVENT" if continuation else "DOWNSTREAM_FIRST_PASSAGE"),))
    void_state = replace(
        replace_cavity(result.state.void_state, updated),
        event_history=result.state.void_state.event_history + ({"event": "CONTINUED_ACCEPTED_EVENT" if continuation else "DOWNSTREAM_FIRST_PASSAGE"},),
    )
    return replace(result.state, void_state=void_state), result, operations


def deterministic_trajectory(*, stop_before_ligament=False):
    state, hole = build_production_void_state(enabled=True)
    cfg = VoidingConfig(enabled=True, promotion_radius_m=5.0e-5)
    rows = [observables(state, "available_site")]
    for label in ("multi_hit_1", "multi_hit_2"):
        tensor = local_site_tensor(state); rates = arrhenius_rates(cfg, temperature_K=900.0, stress_tensor_Pa=tensor)
        site = state.void_state.sites[0]
        dt = max(site.birth.threshold - site.birth.accumulated, 0.0) / (rates["birth_s"] * site.candidate_weight)
        void_state, events = advance_site(state.void_state, site.site_id, dt, rates=rates)
        state = equilibrate_fixed_load_with_production_fem(replace(state, void_state=void_state))
        rows.append({**observables(state, label), "local_tensor_Pa": tensor.tolist(), "rates": rates, "events": events})
    tensor = local_site_tensor(state)
    rates = arrhenius_rates(cfg, temperature_K=900.0, stress_tensor_Pa=tensor)
    site = state.void_state.sites[0]
    dt = site.stabilization.threshold / rates["stabilization_s"]
    void_state, events = advance_site(state.void_state, site.site_id, dt, rates=rates)
    state = equilibrate_fixed_load_with_production_fem(replace(state, void_state=void_state))
    rows.append({**observables(state, "stabilization"), "local_tensor_Pa": tensor.tolist(), "rates": rates, "events": events})
    state = replace(state, void_state=create_subgrid_cavity(state.void_state, "site-1", 2.5e-5))
    state = equilibrate_fixed_load_with_production_fem(state); rows.append(observables(state, "subgrid_void"))
    grown = grow_cavity_2d(state.void_state.cavities[0], 2.5e-5)
    state = replace(state, void_state=replace_cavity(state.void_state, grown))
    state = equilibrate_fixed_load_with_production_fem(state); rows.append(observables(state, "subgrid_growth"))
    promoted = promote_cavity(state.void_state, grown.cavity_id, cfg.promotion_radius_m)
    operations = []
    state = remesh_cavity(state, hole, promoted, "promotion", operations)
    rows.append({**observables(state, "geometric_promotion"), "executed_operations": operations})
    grown_hole = _grow_hole_boundary(hole, 5.5e-5)
    cavity = grow_cavity_2d(state.void_state.cavities[0], 0.5e-5)
    state = remesh_cavity(state, grown_hole, replace_cavity(state.void_state, cavity), "resolved-growth")
    rows.append(observables(state, "resolved_growth"))
    if stop_before_ligament:
        return state, rows
    state, result = ligament_transaction(state)
    rows.append({**observables(state, "ligament_rupture"), "energy_release_J_per_m": result.energy_release_J_per_m})
    rows.append(observables(state, "connected_topology"))
    tensor, boundary_elements = cavity_boundary_tensor(state)
    rates = arrhenius_rates(cfg, temperature_K=900.0, stress_tensor_Pa=tensor)
    downstream_hazard = rates["birth_s"] * 1.0e-12
    rows.append({**observables(state, "downstream_first_passage"), "direct_cavity_boundary_tensor_Pa": tensor.tolist(),
                 "cavity_boundary_element_ids": boundary_elements,
                 "cleavage_rate_s": rates["birth_s"], "integrated_hazard": downstream_hazard})
    state, result, operations = downstream_front_transaction(state)
    rows.append({**observables(state, "new_graph_front"), "executed_operations": operations,
                 "energy_release_J_per_m": result.energy_release_J_per_m})
    state, result, operations = downstream_front_transaction(state, continuation=True)
    rows.append({**observables(state, "continued_accepted_event"), "executed_operations": operations,
                 "energy_release_J_per_m": result.energy_release_J_per_m})
    return state, rows


def natural_trajectory(seed=3621, steps=6):
    state, _ = build_production_void_state(enabled=True, stochastic=True, seed=seed)
    cfg = VoidingConfig(enabled=True)
    rows = []
    for step in range(steps):
        tensor = local_site_tensor(state); rates = arrhenius_rates(cfg, temperature_K=900.0, stress_tensor_Pa=tensor)
        state = equilibrate_fixed_load_with_production_fem(state)
        void_state, events = advance_site(state.void_state, "site-1", 1.0e-12, rates=rates)
        state = replace(state, void_state=void_state)
        rows.append({**observables(state, f"natural_step_{step+1}"), "local_tensor_Pa": tensor.tolist(), "rates": rates, "events": events})
    return state, rows
