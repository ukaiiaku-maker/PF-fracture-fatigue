from dataclasses import replace
import math

import numpy as np

from arrhenius_fracture.branch_cluster_v11 import create_unresolved_branch_cluster
from arrhenius_fracture.directional_competition_v11 import (
    DirectionalCompetitionState, DirectionalRate, commit_directional_interval,
    preview_directional_interval,
)
from arrhenius_fracture.live_topology_kernel_v11 import (
    LiveTopologyRequest, PROVIDER_ID, evaluate_exact_topology,
)
from arrhenius_fracture.production_step_loop_v11 import (
    AcceptedStepContext, advance_accepted_step,
)
from arrhenius_fracture.topology_transaction_v11 import (
    LiveFEMTopologyState, TopologyArm, apply_sharp_wake_trial_geometry,
    execute_topology_trial, extend_network_arm,
)
from tests.test_live_topology_kernel_v11 import live_straight_request


def test_correlated_A12_uses_real_live_fem_and_updates_shared_state_once():
    request = live_straight_request(0.0)
    base_live = evaluate_exact_topology(request)
    candidates = request.candidates_by_tip[request.crack_network.active_tip_ids[0]][:2]
    competition = DirectionalCompetitionState.initialize(candidates, global_hazard_seed=3621)
    competition = replace(competition, hazard_states=tuple(
        commit_directional_interval(
            hazard,
            preview_directional_interval(
                hazard, lambda_per_s=1.0, start_time_s=0.0, duration_s=1.0,
            ),
        )
        for hazard in competition.hazard_states
    ))
    state = LiveFEMTopologyState(
        mesh=request.mesh, boundary=request.boundary, damage=request.damage,
        displacement=np.asarray(base_live["base_equilibrium"]["displacement"]),
        ep_gp=request.ep_gp, rho_gp=request.rho_gp,
        elasticity_D=request.elasticity_D, material=request.material,
        cohesive_network=None, crack_network=request.crack_network,
        competition=competition, tip_process_state={"shared": True},
        junction_process_state={}, energy_ledgers={}, rng_state={"seed": 3621},
        event_counters={"topology_actions": 0, "shared_state_updates": 0},
        stored_energy_J_per_m=base_live["base_equilibrium"]["recoverable_potential_energy_J_per_m"],
    )
    candidate_map = {item.candidate_id: item for item in candidates}
    solve_providers = []

    def trial_action(accepted, proposal):
        network = accepted.crack_network
        cluster = None
        if proposal.action_type == "two_arm":
            network, cluster = create_unresolved_branch_cluster(
                network, parent_branch_id=network.primary_branch_id,
                candidate_ids=proposal.member_candidate_ids, event_index=1,
                shared_process_state={"shared": True},
                conserved_ledgers={name: 0.0 for name in (
                    "retained", "mobile", "escaped", "recovered", "stored_energy",
                    "emission_work", "unconsumed_action",
                )},
            )
        arms = []
        for candidate_id in proposal.member_candidate_ids:
            candidate = candidate_map[candidate_id]
            branch_id = network.primary_branch_id if cluster is None else next(
                item for item in cluster.arm_branch_ids
                if network.branch(item).local_state["candidate_id"] == candidate_id
            )
            start = network.branch(branch_id).tip
            length = 5.0e-6
            end = (start[0] + length * candidate.direction_xy[0], start[1] + length * candidate.direction_xy[1])
            arms.append(TopologyArm(candidate_id, branch_id, start, end, length, 0.0))

        def geometry(trial, realized_arms):
            realized = network
            for arm in realized_arms:
                realized = extend_network_arm(realized, arm)
            trial = replace(
                trial, crack_network=realized,
                junction_process_state=({"cluster": cluster} if cluster else {}),
            )
            return apply_sharp_wake_trial_geometry(
                trial, realized_arms, kill_radius_m=0.5 * request.mesh.hbar_tip,
            )

        def equilibrate(trial):
            by_tip = {}
            for tip_id in trial.crack_network.active_tip_ids:
                physical = trial.crack_network.branch(tip_id).local_state.get("candidate_id")
                by_tip[tip_id] = (
                    (candidate_map[physical],) if physical in candidate_map else candidates
                )
            live_request = replace(
                request, displacement=trial.displacement, damage=trial.damage,
                crack_network=trial.crack_network, candidates_by_tip=by_tip,
                cluster_frame={"mode": "shared_unresolved_cluster"} if cluster else {},
            )
            live = evaluate_exact_topology(live_request)
            solve_providers.append(live["kernel_provider_id"])
            return replace(
                trial, displacement=np.asarray(live["base_equilibrium"]["displacement"]),
                stored_energy_J_per_m=live["base_equilibrium"]["recoverable_potential_energy_J_per_m"],
            )

        return execute_topology_trial(
            accepted, proposal, arms, apply_trial_geometry=geometry,
            equilibrate_fixed_load=equilibrate,
            network_geometry_already_realized=True,
        )

    shared_updates = []
    result = advance_accepted_step(
        state, AcceptedStepContext(1, 1.0, 0.0, "accepted-A12"),
        correlation_interval_s=1.0e-6,
        solve_accepted=lambda current, context: current,
        evaluate_directional_rates=lambda current, context: tuple(
            DirectionalRate(c.candidate_id, 0.0, 1.0, 1.0, 1.0, c.gamma_rel)
            for c in candidates
        ),
        trial_action=trial_action,
        update_shared_state_once=lambda committed, context, proposal: (
            shared_updates.append(proposal.action_id if proposal else None)
            or replace(committed, event_counters={
                **committed.event_counters,
                "shared_state_updates": committed.event_counters["shared_state_updates"] + 1,
            })
        ),
    )
    selected = next(item for item in result.trials if item.selected)
    assert selected.proposal.action_type == "two_arm"
    assert len(result.state.crack_network.active_tip_ids) == 2
    assert set(result.state.competition.consumed_event_ids) == set(selected.proposal.member_event_ids)
    assert result.state.event_counters["shared_state_updates"] == 1
    assert len(shared_updates) == 1
    assert solve_providers and set(solve_providers) == {PROVIDER_ID}
