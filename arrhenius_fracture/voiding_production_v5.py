"""Executable one-void trajectory owned by the V12 production FEM state."""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree

from .crack_network_v11 import CrackBranchState, CrackNetworkState, ROOT_BRANCH_ID
from .directional_competition_v11 import (
    DirectionalCompetitionState, commit_directional_interval,
    construct_action_proposals, preview_directional_interval,
    tungsten_cleavage_candidates,
)
from .explicit_cavity_v5 import build_explicit_hole_mesh, fill_explicit_hole_mesh
from .mesh import rebuild_tri_mesh
from .fem import assemble_mechanics, plane_strain_D
from .sharp_front import (
    FrontConfig, FrontEngine, default_cleavage_barrier,
    default_emission_barrier, make_emergent_config,
)
from .hazard_energy_event_gate_v10230 import hazard_resistance_J_per_m2
from .sharp_wake_backend_v12 import V12_MODEL_ID
from .topology_transaction_v11 import (
    LiveFEMTopologyState, TopologyArm, apply_v12_production_trial_geometry,
    complete_accepted_state_fingerprint, equilibrate_fixed_load_with_production_fem,
    execute_topology_trial, initialize_mechanically_separating_v12,
    remesh_mechanically_separating_v12,
)
from .voiding_v5 import (
    Cavity2D, HazardClock, ProductionVoidState, VoidPhase, VoidSite, VoidingConfig,
    advance_site, arrhenius_rates, create_subgrid_cavity, grow_cavity_2d,
    grow_cavity_from_rate,
    promote_cavity, replace_cavity,
)

SCHEMA = "v12.production-one-void-trajectory/5"


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
        CrackNetworkState.one_tip((start, tip)), DirectionalCompetitionState.initialize(
            tungsten_cleavage_candidates(theta_deg=0.0), global_hazard_seed=seed,
        ),
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
        "length_ledgers": {} if state.void_state is None else dict(state.void_state.length_ledgers),
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


def cavity_boundary_tensor(state, *, boundary_node: int | None = None):
    """Return the most tensile resolved tensor on the explicit cavity boundary.

    First passage is local, so circumferential averaging (which can cancel the
    tensile hot spot) is not a valid kinetic input.
    """
    _, _, sigma, *_ = assemble_mechanics(
        state.mesh, state.displacement, state.ep_gp, state.rho_gp, state.damage,
        state.elasticity_D, state.material, cohesive_network=state.cohesive_network,
    )
    cavity = state.void_state.cavities[0]
    center = np.asarray(cavity.center_m)
    radii = np.linalg.norm(np.asarray(state.mesh.nodes) - center, axis=1)
    boundary_nodes = np.flatnonzero(radii <= cavity.radius_m * 1.02)
    selected_nodes = boundary_nodes if boundary_node is None else np.asarray((int(boundary_node),))
    if boundary_node is not None and int(boundary_node) not in set(map(int, boundary_nodes)):
        raise ValueError("requested tensor node is not on the cavity boundary")
    elements = np.flatnonzero(np.any(np.isin(state.mesh.elems, selected_nodes), axis=1))
    tensors = np.asarray([
        [[sigma[0, element], sigma[2, element]],
         [sigma[2, element], sigma[1, element]]]
        for element in elements
    ])
    selected = int(np.argmax(np.linalg.eigvalsh(tensors)[:, -1]))
    return tensors[selected], tuple(map(int, elements))


def _project_fields(state, mesh):
    old_nodes = np.asarray(state.mesh.nodes); new_nodes = np.asarray(mesh.nodes)
    _, node_parent = cKDTree(old_nodes).query(new_nodes)
    old_centers = old_nodes[state.mesh.elems].mean(axis=1)
    new_centers = new_nodes[mesh.elems].mean(axis=1)
    _, element_parent = cKDTree(old_centers).query(new_centers)
    def p1(values):
        source = np.asarray(values, dtype=float)
        projected = np.asarray(LinearNDInterpolator(old_nodes, source)(new_nodes))
        missing = ~np.all(np.isfinite(projected), axis=1) if projected.ndim == 2 else ~np.isfinite(projected)
        projected[missing] = source[node_parent[missing]]
        return projected

    ep = np.asarray(state.ep_gp, dtype=float)[:, element_parent]
    rho = np.asarray(state.rho_gp, dtype=float)[element_parent]
    new_area = np.asarray(mesh.area_e, dtype=float)
    old_area = np.asarray(state.mesh.area_e, dtype=float)
    for component in range(ep.shape[0]):
        source_integral = float(np.sum(np.asarray(state.ep_gp)[component] * old_area))
        projected_integral = float(np.sum(ep[component] * new_area))
        if abs(projected_integral) > 1.0e-300:
            ep[component] *= source_integral / projected_integral
    source_rho = float(np.sum(np.asarray(state.rho_gp) * old_area))
    projected_rho = float(np.sum(rho * new_area))
    if abs(projected_rho) > 1.0e-300:
        rho *= source_rho / projected_rho
    return {
        "damage": np.clip(p1(np.asarray(state.damage)), 0.0, 1.0),
        "displacement": p1(np.asarray(state.displacement).reshape(-1, 2)).reshape(-1),
        "ep_gp": ep,
        "rho_gp": rho,
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
        "plastic_integral_error": float(np.max(np.abs(
            np.sum(fields["ep_gp"] * hole.mesh.area_e[None, :], axis=1)
            - np.sum(state.ep_gp * state.mesh.area_e[None, :], axis=1)
        ))),
        "density_integral_error": float(abs(
            np.sum(fields["rho_gp"] * hole.mesh.area_e)
            - np.sum(state.rho_gp * state.mesh.area_e)
        )),
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


def _complete_next_clock(state, stress_tensor_Pa, *, start_time=0.0, temperature_K=900.0):
    """Advance all directional clocks through one common earliest-event time."""
    material = state.material
    engine = FrontEngine(
        FrontConfig(), default_cleavage_barrier(), default_emission_barrier(material.b),
        material.G, material.nu, material.b,
    )
    stress = np.asarray(stress_tensor_Pa, dtype=float).reshape(2, 2)
    rates = []
    hazards = list(state.competition.hazard_states)
    crossing_times = []
    for candidate, hazard in zip(state.competition.candidates, hazards):
        normal = np.asarray(candidate.normal_xy, dtype=float)
        resolved_opening = max(float(normal @ stress @ normal), 0.0)
        raw_rate, _, barrier = engine.lambda_cleave(resolved_opening, temperature_K)
        zero_tolerance = 1.0e-12 * max(float(np.linalg.norm(stress, ord=2)), 1.0)
        rate = 0.0 if resolved_opening <= zero_tolerance else max(float(raw_rate), 0.0)
        remaining = max(hazard.current_threshold_action - hazard.action, 0.0)
        crossing = math.inf if rate <= 0.0 else remaining / rate
        crossing_times.append(crossing)
        rates.append({"candidate_id": candidate.candidate_id, "rate_s": rate,
                      "resolved_opening_stress_Pa": resolved_opening,
                      "hazard_barrier_J": barrier, "crossing_time_s": crossing})
    duration = min(crossing_times)
    if not math.isfinite(duration):
        raise RuntimeError("production cleavage clock cannot reach first passage")
    for index, (hazard, rate) in enumerate(zip(hazards, rates)):
        hazards[index] = commit_directional_interval(
            hazard, preview_directional_interval(
                hazard, lambda_per_s=rate["rate_s"], start_time_s=start_time,
                duration_s=duration,
            ),
        )
        rate["common_advance_duration_s"] = duration
        rate["winner"] = abs(rate["crossing_time_s"] - duration) <= 1.0e-12 * max(1.0, duration)
    return replace(state, competition=replace(state.competition, hazard_states=tuple(hazards))), rates


def ligament_transaction(state, *, failure_stage=None, operation_log=None):
    cavity = state.void_state.cavities[0]
    start = state.crack_network.branch(ROOT_BRANCH_ID).tip
    nodes = np.asarray(state.mesh.nodes)
    center = np.asarray(cavity.center_m)
    radii = np.linalg.norm(nodes - center, axis=1)
    boundary_nodes = np.flatnonzero(radii <= cavity.radius_m * 1.02)
    # The selected theta=0 cleavage ray is horizontal.  Its first intersection
    # with the actual polygonal free boundary is the leftmost boundary vertex;
    # no unrelated nearest-volume node is permitted.
    node = int(boundary_nodes[np.argmin(nodes[boundary_nodes, 0])])
    end = tuple(map(float, state.mesh.nodes[node]))
    if abs(end[1] - start[1]) > 1.0e-14 or end[0] <= start[0]:
        raise RuntimeError("no exact selected-direction cavity-boundary intersection")
    state, cleavage_audit = _complete_next_clock(state, cavity_boundary_tensor(state)[0])
    proposal = next(item for item in construct_action_proposals(
        state.competition.hazard_states, correlation_interval_s=0.0,
    ) if item.action_type == "one_arm")
    candidate = next(item for item in state.competition.candidates if item.candidate_id == proposal.member_candidate_ids[0])
    engine = FrontEngine(FrontConfig(), default_cleavage_barrier(), default_emission_barrier(state.material.b),
                         state.material.G, state.material.nu, state.material.b)
    winner = next(item for item in cleavage_audit if item["winner"])
    barrier = winner["hazard_barrier_J"]
    resistance = hazard_resistance_J_per_m2(
        barrier_J=barrier, cooperative_hits=engine.f.m_hits,
        burgers_vector_m=engine.b, gamma_relative=candidate.gamma_rel,
    )
    arm = TopologyArm(
        proposal.member_candidate_ids[0], ROOT_BRANCH_ID, start, end, math.dist(start, end),
        resistance * math.dist(start, end), event_classification="physical_cleavage",
        candidate_direction_xy=candidate.direction_xy, first_intersection_xy_m=end,
    )
    operations = operation_log if operation_log is not None else []
    def inject(stage, current):
        operations.append(stage)
        if stage == failure_stage: raise RuntimeError("injected:" + stage)
    def geometry(trial, arms):
        realized = apply_v12_production_trial_geometry(
            trial, arms, source_commit=_head(), configuration={"event": "CRACK_TO_VOID_LIGAMENT"},
            transaction_identity="ligament", failure_injector=inject,
        )
        connected = replace(
            realized.void_state.cavities[0], phase=VoidPhase.CONNECTED_VOID,
            lineage=realized.void_state.cavities[0].lineage + ("CRACK_TO_VOID_LIGAMENT",),
        )
        void_state = replace_cavity(realized.void_state, connected)
        inject("cavity_phase_update", replace(realized, void_state=void_state))
        ledgers = dict(void_state.length_ledgers)
        ligament_length = math.dist(start, end)
        increments = {
            "fractured_ligament_length_m": ligament_length,
            "active_front_coordinate_advance_m": ligament_length,
            "projected_fractured_length_m": end[0] - start[0],
            "projected_front_advance_m": end[0] - start[0],
            "preexisting_void_free_span_m": 2.0 * cavity.radius_m,
            "projected_free_span_m": 2.0 * cavity.radius_m,
            "connected_free_surface_extent_m": math.pi * cavity.radius_m,
        }
        for name, increment in increments.items():
            ledgers[name] += increment
            void_state = replace(void_state, length_ledgers=dict(ledgers))
            inject("length_ledger_update:" + name, replace(realized, void_state=void_state))
        void_state = replace(
            void_state,
            event_history=void_state.event_history + ({"event": "CRACK_TO_VOID_LIGAMENT"},),
        )
        realized = replace(realized, void_state=void_state)
        _, boundary_elements = cavity_boundary_tensor(realized)
        if not boundary_elements:
            raise RuntimeError("connected free surface is not certified")
        inject("connected_surface_certification", realized)
        return realized
    result = execute_topology_trial(
        state, proposal, (arm,), apply_trial_geometry=geometry,
        equilibrate_fixed_load=equilibrate_fixed_load_with_production_fem,
        network_geometry_already_realized=True, failure_injector=inject,
    )
    if not result.accepted: raise RuntimeError("ligament event rejected")
    return result.state, result


def downstream_front_transaction(state, *, continuation=False):
    child_id = "void-front-1"
    center = np.asarray(state.void_state.cavities[0].center_m)
    nodes = np.asarray(state.mesh.nodes)
    boundary_radii = np.linalg.norm(nodes - center, axis=1)
    cavity_nodes = np.flatnonzero(boundary_radii <= state.void_state.cavities[0].radius_m * 1.02)
    right_boundary = int(cavity_nodes[np.argmax(nodes[cavity_nodes, 0])])
    tensor, boundary_elements = cavity_boundary_tensor(state, boundary_node=right_boundary)
    state, cleavage_audit = _complete_next_clock(
        state, tensor, start_time=2.0 if continuation else 1.0,
    )
    proposal = next(item for item in construct_action_proposals(
        state.competition.hazard_states, correlation_interval_s=0.0,
    ) if item.action_type == "one_arm")
    if continuation:
        start = state.crack_network.branch(child_id).tip
        candidates = np.flatnonzero(
            (nodes[:, 0] > start[0] + 1.0e-8)
            & (np.abs(nodes[:, 1] - start[1]) <= 1.0e-10)
        )
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
        candidates = np.flatnonzero(
            (nodes[:, 0] > start[0] + 1.0e-8)
            & (np.abs(nodes[:, 1] - start[1]) <= 1.0e-10)
        )
        end_node = int(candidates[np.argmin(np.linalg.norm(nodes[candidates] - (np.asarray(start) + [1.5e-4, 0.0]), axis=1))])
    end = tuple(map(float, nodes[end_node]))
    candidate = next(item for item in state.competition.candidates if item.candidate_id == proposal.member_candidate_ids[0])
    engine = FrontEngine(FrontConfig(), default_cleavage_barrier(), default_emission_barrier(state.material.b),
                         state.material.G, state.material.nu, state.material.b)
    winner = next(item for item in cleavage_audit if item["winner"])
    barrier = winner["hazard_barrier_J"]
    resistance = hazard_resistance_J_per_m2(
        barrier_J=barrier, cooperative_hits=engine.f.m_hits,
        burgers_vector_m=engine.b, gamma_relative=candidate.gamma_rel,
    )
    arm = TopologyArm(
        proposal.member_candidate_ids[0], child_id, start, end, math.dist(start, end),
        resistance * math.dist(start, end), event_classification="physical_cleavage",
        candidate_direction_xy=candidate.direction_xy, first_intersection_xy_m=end,
    )
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
    ledgers = dict(result.state.void_state.length_ledgers)
    advance = math.dist(start, end)
    ledgers["ordinary_crack_fractured_length_m"] += advance
    ledgers["active_front_coordinate_advance_m"] += advance
    ledgers["projected_fractured_length_m"] += end[0] - start[0]
    ledgers["projected_front_advance_m"] += end[0] - start[0]
    void_state = replace(
        replace_cavity(result.state.void_state, updated),
        event_history=result.state.void_state.event_history + ({"event": "CONTINUED_ACCEPTED_EVENT" if continuation else "DOWNSTREAM_FIRST_PASSAGE"},),
        length_ledgers=ledgers,
    )
    return replace(result.state, void_state=void_state), result, operations, {
        "tensor_Pa": tensor.tolist(), "boundary_element_ids": boundary_elements,
        "boundary_position_m": list(map(float, nodes[right_boundary])),
        "surface_normal_xy": [1.0, 0.0], "surface_tangent_xy": [0.0, 1.0],
        "candidate_id": proposal.member_candidate_ids[0],
        "cavity_id": cavity.cavity_id, "cleavage": cleavage_audit,
    }


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
    tensor = local_site_tensor(state)
    rates = arrhenius_rates(cfg, temperature_K=900.0, stress_tensor_Pa=tensor)
    growth_dt = 2.5e-5 / (cfg.radial_growth_scale_m * rates["series_limited_growth_s"])
    grown = grow_cavity_from_rate(
        state.void_state.cavities[0], rates=rates, dt_s=growth_dt,
        radial_growth_scale_m=cfg.radial_growth_scale_m,
    )
    state = replace(state, void_state=replace_cavity(state.void_state, grown))
    state = equilibrate_fixed_load_with_production_fem(state)
    rows.append({**observables(state, "subgrid_growth"), "rates": rates, "growth_dt_s": growth_dt})
    promoted = promote_cavity(state.void_state, grown.cavity_id, cfg.promotion_radius_m)
    operations = []
    state = remesh_cavity(state, hole, promoted, "promotion", operations)
    rows.append({**observables(state, "geometric_promotion"), "executed_operations": operations})
    grown_hole = _grow_hole_boundary(hole, 5.5e-5)
    tensor = cavity_boundary_tensor(state)[0]
    rates = arrhenius_rates(cfg, temperature_K=900.0, stress_tensor_Pa=tensor)
    growth_dt = 0.5e-5 / (cfg.radial_growth_scale_m * rates["series_limited_growth_s"])
    cavity = grow_cavity_from_rate(
        state.void_state.cavities[0], rates=rates, dt_s=growth_dt,
        radial_growth_scale_m=cfg.radial_growth_scale_m,
    )
    state = remesh_cavity(state, grown_hole, replace_cavity(state.void_state, cavity), "resolved-growth")
    rows.append({**observables(state, "resolved_growth"), "rates": rates, "growth_dt_s": growth_dt})
    if stop_before_ligament:
        return state, rows
    state, result = ligament_transaction(state)
    rows.append({
        **observables(state, "ligament_rupture"),
        "energy_release_J_per_m": result.energy_release_J_per_m,
        "hazard_dissipation_J_per_m": result.hazard_dissipation_J_per_m,
        "energy_margin_J_per_m": result.energy_margin_J_per_m,
        "event_classification": "physical_cleavage",
    })
    rows.append(observables(state, "connected_topology"))
    tensor, boundary_elements = cavity_boundary_tensor(state)
    rates = arrhenius_rates(cfg, temperature_K=900.0, stress_tensor_Pa=tensor)
    rows.append({**observables(state, "downstream_surface_probe"), "direct_cavity_boundary_tensor_Pa": tensor.tolist(),
                 "cavity_boundary_element_ids": boundary_elements,
                 "void_birth_rate_s": rates["birth_s"],
                 "classification": "PRE_CLEAVAGE_SURFACE_PROBE"})
    state, result, operations, causal = downstream_front_transaction(state)
    rows.append({**observables(state, "new_graph_front"), "executed_operations": operations,
                 "energy_release_J_per_m": result.energy_release_J_per_m,
                 "causal_first_passage": causal})
    state, result, operations, causal = downstream_front_transaction(state, continuation=True)
    rows.append({**observables(state, "continued_accepted_event"), "executed_operations": operations,
                 "energy_release_J_per_m": result.energy_release_J_per_m,
                 "causal_first_passage": causal})
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
