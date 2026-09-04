"""Executable, evidence-producing V11/V12 bounded production trajectories."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from .checkpoint_v11 import restore_checkpoint, write_checkpoint
from .conforming_crack_oracle_v12 import build_matched_crack_parent
from .crack_network_v11 import CrackNetworkState, ROOT_BRANCH_ID
from .branch_cluster_v11 import create_unresolved_branch_cluster
from .directional_competition_v11 import (
    DirectionalCompetitionState,
    DirectionalHazardState,
    commit_directional_interval,
    construct_action_proposals,
    preview_directional_interval,
    tungsten_cleavage_candidates,
)
from .fem import plane_strain_D
from .sharp_front import (
    FrontConfig, FrontEngine, default_cleavage_barrier,
    default_emission_barrier, make_emergent_config,
)
from .hazard_energy_event_gate_v10230 import hazard_resistance_J_per_m2
from .sharp_wake_backend_v12 import V11_MODEL_ID, V12_MODEL_ID, select_sharp_wake_model
from .topology_transaction_v11 import (
    LiveFEMTopologyState,
    TopologyArm,
    apply_causal_sharp_wake_trial_geometry,
    apply_v12_production_trial_geometry,
    complete_accepted_state_fingerprint,
    equilibrate_fixed_load_with_production_fem,
    execute_topology_trial,
    initialize_mechanically_separating_v12,
)

SCHEMA = "v12.executable-production-evidence/2"


def _competition(seed: int) -> DirectionalCompetitionState:
    candidates = tungsten_cleavage_candidates(theta_deg=45.0)
    state = DirectionalCompetitionState.initialize(candidates, global_hazard_seed=seed)
    hazards = []
    for index, candidate in enumerate(state.candidates):
        initial = DirectionalHazardState(candidate.candidate_id)
        duration = 1.0 if index == 0 else 0.375
        hazards.append(commit_directional_interval(
            initial,
            preview_directional_interval(initial, lambda_per_s=1.0,
                                         start_time_s=0.0, duration_s=duration),
        ))
    return replace(state, hazard_states=tuple(hazards))


def _two_arm_competition(seed: int) -> DirectionalCompetitionState:
    state = DirectionalCompetitionState.initialize(
        tungsten_cleavage_candidates(theta_deg=45.0), global_hazard_seed=seed,
    )
    hazards = tuple(commit_directional_interval(
        DirectionalHazardState(candidate.candidate_id),
        preview_directional_interval(
            DirectionalHazardState(candidate.candidate_id), lambda_per_s=1.0,
            start_time_s=0.0, duration_s=1.0,
        ),
    ) for candidate in state.candidates)
    return replace(state, hazard_states=hazards)


def build_loaded_state(model: str, *, seed: int = 3621) -> LiveFEMTopologyState:
    selected = select_sharp_wake_model(model)
    parent = build_matched_crack_parent(
        8.0e-4, 8.0e-4, (2.0e-4, 0.0), (5.0e-4, 0.0), 25.0e-6,
    )
    mesh = parent.mesh
    cfg = make_emergent_config()
    displacement = np.zeros(mesh.ndof)
    displacement[2 * np.asarray(parent.boundary.top_nodes) + 1] = 2.0e-7
    displacement[2 * np.asarray(parent.boundary.bot_nodes) + 1] = -2.0e-7
    state = LiveFEMTopologyState(
        mesh=mesh,
        boundary=parent.boundary,
        damage=np.zeros(mesh.nn),
        displacement=displacement,
        ep_gp=np.vstack((
            2.0e-5 * np.sin(mesh.nodes[mesh.elems].mean(axis=1)[:, 0] / 8.0e-4 * np.pi),
            1.0e-5 * np.cos(mesh.nodes[mesh.elems].mean(axis=1)[:, 1] / 8.0e-4 * np.pi),
            0.5e-5 * np.ones(mesh.ne),
        )),
        rho_gp=1.0e12 * (1.0 + 0.1 * mesh.nodes[mesh.elems].mean(axis=1)[:, 0] / 8.0e-4),
        elasticity_D=plane_strain_D(cfg.material),
        material=cfg.material,
        cohesive_network=None,
        crack_network=CrackNetworkState.one_tip(((2.0e-4, 0.0), (5.0e-4, 0.0))),
        competition=_competition(seed),
        tip_process_state={"retained": 7.0, "mobile": 2.0, "hazard_clock_incomplete": True},
        junction_process_state={"crack_representation": V11_MODEL_ID},
        energy_ledgers={"emission_work": 3.0, "source_work": 1.25},
        rng_state=np.random.default_rng(seed).bit_generator.state,
        event_counters={"topology_actions": 0, "accepted_steps": 1, "mesh_generation": 0},
        stored_energy_J_per_m=0.0,
        sharp_wake_model_id=V11_MODEL_ID,
        checkpoint_generation=1,
    )
    if selected == V12_MODEL_ID:
        state = initialize_mechanically_separating_v12(
            state,
            source_commit=_git_head(),
            configuration={"entry_point": "v12_production_driver", "model": selected},
        )
    return equilibrate_fixed_load_with_production_fem(state)


def _git_head() -> str:
    return subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=Path(__file__).resolve().parents[1], text=True,
    ).strip()


def _observables(state: LiveFEMTopologyState) -> dict[str, Any]:
    support = state.v12_support_state
    return {
        "fingerprint": complete_accepted_state_fingerprint(state),
        "model": state.sharp_wake_model_id,
        "mesh_nodes": int(state.mesh.nn),
        "mesh_elements": int(state.mesh.ne),
        "mesh_generation": int(state.event_counters.get("mesh_generation", 0)),
        "graph_length_m": float(state.crack_network.total_physical_crack_length_m),
        "active_tips": list(state.crack_network.active_tip_ids),
        "support_elements": 0 if support is None else len(support.selected_support_elements),
        "damage_l1": float(np.sum(state.damage)),
        "displacement_l2_m": float(np.linalg.norm(state.displacement)),
        "plastic_history_l2": float(np.linalg.norm(state.ep_gp)),
        "density_l2_m2": float(np.linalg.norm(state.rho_gp)),
        "reaction_N_per_m": float(state.energy_ledgers.get("latest_reaction_N_per_m", 0.0)),
        "residual_l2_N_per_m": float(state.energy_ledgers.get("latest_residual_l2_N_per_m", 0.0)),
        "stored_energy_J_per_m": float(state.stored_energy_J_per_m),
        "competition_consumed": list(state.competition.consumed_event_ids),
        "hazards": [float(item.residual_action) for item in state.competition.hazard_states],
        "thresholds": [float(item.current_threshold_action) for item in state.competition.hazard_states],
        "rng_state": state.rng_state,
        "tip_process_state": dict(state.tip_process_state),
        "event_counters": dict(state.event_counters),
    }


def execute_event(
    state: LiveFEMTopologyState,
    end_xy_m: tuple[float, float],
    *,
    transaction_identity: str,
    failure_stage: str | None = None,
    operation_log: list[str] | None = None,
) -> tuple[LiveFEMTopologyState, dict[str, Any]]:
    proposal = next(item for item in construct_action_proposals(
        state.competition.hazard_states, correlation_interval_s=0.0,
    ) if item.action_type == "one_arm")
    start = state.crack_network.branch(ROOT_BRANCH_ID).tip
    length = float(np.linalg.norm(np.asarray(end_xy_m) - np.asarray(start)))
    arm = TopologyArm(
        proposal.member_candidate_ids[0], ROOT_BRANCH_ID, start, end_xy_m, length, 0.0,
    )
    operations: list[str] = []

    def inject(stage: str, trial: LiveFEMTopologyState) -> None:
        operations.append(stage)
        if operation_log is not None:
            operation_log.append(stage)
        if stage == failure_stage:
            raise RuntimeError("injected:" + stage)

    if state.sharp_wake_model_id == V12_MODEL_ID:
        geometry = lambda trial, arms: apply_v12_production_trial_geometry(
            trial, arms,
            source_commit=_git_head(),
            configuration={"entry_point": "v12_production_driver", "model": V12_MODEL_ID},
            transaction_identity=transaction_identity,
            failure_injector=inject,
        )
        realized = True
    else:
        geometry = apply_causal_sharp_wake_trial_geometry
        realized = False
    before = _observables(state)
    result = execute_topology_trial(
        state, proposal, (arm,),
        apply_trial_geometry=geometry,
        equilibrate_fixed_load=equilibrate_fixed_load_with_production_fem,
        network_geometry_already_realized=realized,
        failure_injector=inject,
    )
    if not result.accepted:
        raise RuntimeError("production event rejected: " + str(result.rejection_reason))
    after = _observables(result.state)
    return result.state, {
        "initial": before,
        "final": after,
        "operations": operations,
        "energy_release_J_per_m": result.energy_release_J_per_m,
        "hazard_dissipation_J_per_m": result.hazard_dissipation_J_per_m,
        "energy_margin_J_per_m": result.energy_margin_J_per_m,
    }


def execute_physical_two_arm_event(
    state: LiveFEMTopologyState, *, arm_length_m: float = 25.0e-6 * 2.0 ** 0.5,
    event_stress_Pa: float = 40.0e9, event_temperature_K: float = 900.0,
    failure_stage: str | None = None, operation_log: list[str] | None = None,
) -> tuple[LiveFEMTopologyState, dict[str, Any]]:
    """Execute a genuine correlated two-candidate branch transaction.

    Dissipation is calculated from the active production cleavage barrier and
    converted by the existing hazard-energy law; it is not an invented constant.
    """
    if state.sharp_wake_model_id != V12_MODEL_ID:
        raise ValueError("physical two-arm production event requires V12 mechanics")
    state = replace(state, competition=_two_arm_competition(state.competition.global_hazard_seed))
    proposals = construct_action_proposals(
        state.competition.hazard_states, correlation_interval_s=1.0e-12,
    )
    proposal = next(item for item in proposals if item.action_type == "two_arm")
    candidates = {item.candidate_id: item for item in state.competition.candidates}
    material = state.material
    engine = FrontEngine(
        FrontConfig(), default_cleavage_barrier(), default_emission_barrier(material.b),
        material.G, material.nu, material.b,
    )
    _, _, barrier_J = engine.lambda_cleave(event_stress_Pa, event_temperature_K)
    network, cluster = create_unresolved_branch_cluster(
        state.crack_network, parent_branch_id=ROOT_BRANCH_ID,
        candidate_ids=proposal.member_candidate_ids, event_index=1,
        shared_process_state=dict(state.tip_process_state),
        conserved_ledgers={
            "retained": 7.0, "mobile": 2.0, "stored_energy": state.stored_energy_J_per_m,
            "emission_work": 3.0, "unconsumed_action": 2.0,
        },
    )
    state = replace(
        state, crack_network=network,
        junction_process_state={**state.junction_process_state, "branch_cluster": asdict(cluster)},
    )
    start = network.branch(cluster.arm_branch_ids[0]).tip
    arms = []
    candidate_to_branch = dict(zip(proposal.member_candidate_ids, cluster.arm_branch_ids))
    for candidate_id in proposal.member_candidate_ids:
        direction = np.asarray(candidates[candidate_id].direction_xy, dtype=float)
        direction /= np.linalg.norm(direction)
        endpoint = tuple((np.asarray(start) + arm_length_m * direction).tolist())
        resistance = hazard_resistance_J_per_m2(
            barrier_J=barrier_J, cooperative_hits=engine.f.m_hits,
            burgers_vector_m=engine.b,
            gamma_relative=candidates[candidate_id].gamma_rel,
        )
        hazard_dissipation_J_per_m = resistance * arm_length_m
        arms.append(TopologyArm(
            candidate_id, candidate_to_branch[candidate_id], start, endpoint,
            arm_length_m, hazard_dissipation_J_per_m,
            event_classification="physical_cleavage",
            candidate_direction_xy=tuple(direction.tolist()),
            first_intersection_xy_m=endpoint,
        ))
    operations: list[str] = []

    def inject(stage: str, trial: LiveFEMTopologyState) -> None:
        operations.append(stage)
        if operation_log is not None:
            operation_log.append(stage)
        if stage == failure_stage:
            raise RuntimeError("injected:" + stage)

    result = execute_topology_trial(
        state, proposal, tuple(arms),
        apply_trial_geometry=lambda trial, values: apply_v12_production_trial_geometry(
            trial, values, source_commit=_git_head(),
            configuration={"entry_point": "v12_production_driver", "model": V12_MODEL_ID},
            transaction_identity="physical-two-arm",
            failure_injector=inject,
        ),
        equilibrate_fixed_load=equilibrate_fixed_load_with_production_fem,
        network_geometry_already_realized=True,
        failure_injector=inject,
    )
    return result.state, {
        "accepted": result.accepted,
        "action_type": proposal.action_type,
        "candidate_ids": list(proposal.member_candidate_ids),
        "branch_ids": list(cluster.arm_branch_ids),
        "directions": [list(arm.candidate_direction_xy) for arm in arms],
        "endpoints_m": [list(arm.end_xy_m) for arm in arms],
        "energy_release_J_per_m": result.energy_release_J_per_m,
        "hazard_dissipation_J_per_m": result.hazard_dissipation_J_per_m,
        "energy_margin_J_per_m": result.energy_margin_J_per_m,
        "event_stress_Pa": event_stress_Pa,
        "event_temperature_K": event_temperature_K,
        "hazard_barrier_J": barrier_J,
        "rejection_reason": result.rejection_reason,
        "operations": operations,
    }


def run_trajectory(model: str, geometry: str) -> dict[str, Any]:
    state = build_loaded_state(model)
    initial = _observables(state)
    paths = {
        "straight": ((5.25e-4, 0.0),),
        "sequential": ((5.25e-4, 0.0), (5.50e-4, 0.0)),
        "kink": ((5.25e-4, 2.50e-5),),
        "oblique": ((5.25e-4, -1.25e-5),),
        "refinement": ((5.125e-4, 0.0),),
    }[geometry]
    events = []
    for index, endpoint in enumerate(paths, 1):
        state, event = execute_event(state, endpoint, transaction_identity=f"{geometry}-{index}")
        events.append(event)
        if index < len(paths):
            # Complete the next physical candidate clock through the same state owner.
            hazards = list(state.competition.hazard_states)
            for position, hazard in enumerate(hazards):
                if hazard.candidate_id not in state.competition.consumed_event_ids:
                    hazards[position] = commit_directional_interval(
                        hazard,
                        preview_directional_interval(hazard, lambda_per_s=1.0,
                                                     start_time_s=1.0, duration_s=1.0),
                    )
                    break
            state = replace(state, competition=replace(state.competition, hazard_states=tuple(hazards)))
    return {"case": geometry, "initial": initial, "final": _observables(state), "events": events}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sharp-wake-model", choices=(V11_MODEL_ID, V12_MODEL_ID), default=V11_MODEL_ID)
    parser.add_argument("--trajectory", choices=("straight", "sequential", "kink", "oblique", "refinement"), default="straight")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    row = run_trajectory(args.sharp_wake_model, args.trajectory)
    payload = {"schema": SCHEMA, "git_head": _git_head(), "argv": vars(args) | {"out": str(args.out)}, "row": row}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
