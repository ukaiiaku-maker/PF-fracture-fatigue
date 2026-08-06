from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from arrhenius_fracture.branch_cluster_v11 import create_unresolved_branch_cluster
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.live_topology_runtime_v11 import LiveTopologyRuntime
from arrhenius_fracture.crack_backend import SharpWakeBackend
from arrhenius_fracture.crack_network_v11 import CrackNetworkState, ROOT_BRANCH_ID
from arrhenius_fracture.directional_competition_v11 import (
    DirectionalCompetitionState,
    commit_directional_interval,
    construct_action_proposals,
    preview_directional_interval,
    tungsten_cleavage_candidates,
)
from arrhenius_fracture.topology_transaction_v11 import (
    LiveFEMTopologyState,
    TopologyArm,
    apply_sharp_wake_trial_geometry,
    clip_arm_at_first_intersection,
    execute_topology_trial,
    extend_network_arm,
    mark_coalesced,
)


def competition():
    candidates = tungsten_cleavage_candidates(theta_deg=45.0)
    state = DirectionalCompetitionState.initialize(candidates, global_hazard_seed=3621)
    hazards = tuple(
        commit_directional_interval(
            hazard,
            preview_directional_interval(
                hazard, lambda_per_s=1.0, start_time_s=0.0, duration_s=1.0
            ),
        )
        for hazard in state.hazard_states
    )
    return replace(state, hazard_states=hazards)


def fem_state(comp=None):
    mesh = SimpleNamespace(
        nodes=np.array([[0.0, -1.0], [0.0, 1.0], [2.0, -1.0], [2.0, 1.0]]),
        elems=np.array([[0, 2, 1], [1, 2, 3]]),
        area_e=np.ones(2),
    )
    return LiveFEMTopologyState(
        mesh=mesh, boundary={"fixed": (0, 1)}, damage=np.zeros(4),
        displacement=np.arange(8.0), ep_gp=np.zeros((3, 2)), rho_gp=np.ones(2),
        elasticity_D=np.eye(3), material={"name": "test"}, cohesive_network=None,
        crack_network=CrackNetworkState.one_tip(((0.0, 0.0), (1.0, 0.0))),
        competition=comp or competition(), tip_process_state={"retained": 7.0},
        junction_process_state={}, energy_ledgers={"emission_work": 3.0},
        rng_state={"state": [3, 6, 2, 1]}, event_counters={"topology_actions": 4},
        stored_energy_J_per_m=10.0,
    )


def arm(candidate_id, branch_id=ROOT_BRANCH_ID, end=(1.5, 0.0), dissipation=2.0):
    return TopologyArm(
        candidate_id=candidate_id, branch_id=branch_id,
        start_xy_m=(1.0, 0.0), end_xy_m=end,
        event_reward_m=0.5, hazard_dissipation_J_per_m=dissipation,
    )


def equilibrate_to(energy):
    return lambda state: replace(
        state, displacement=np.asarray(state.displacement) + 0.25,
        stored_energy_J_per_m=energy,
    )


def test_rejected_trial_is_exact_accepted_snapshot_and_consumes_nothing():
    accepted = fem_state()
    proposal = next(p for p in construct_action_proposals(accepted.competition.hazard_states, correlation_interval_s=0.0) if p.action_type == "one_arm")
    trial_arm = arm(proposal.member_candidate_ids[0], dissipation=5.0)
    before_damage = accepted.damage.copy()
    before_u = accepted.displacement.copy()
    result = execute_topology_trial(
        accepted, proposal, (trial_arm,),
        apply_trial_geometry=lambda state, arms: apply_sharp_wake_trial_geometry(state, arms, kill_radius_m=0.2),
        equilibrate_fixed_load=equilibrate_to(9.0),
    )
    assert result.accepted is False
    assert result.state is accepted
    np.testing.assert_array_equal(accepted.damage, before_damage)
    np.testing.assert_array_equal(accepted.displacement, before_u)
    assert accepted.competition.consumed_event_ids == ()
    assert accepted.competition.reservations == ()
    assert accepted.event_counters == {"topology_actions": 4}


def test_one_arm_transaction_commits_one_reward_and_one_event():
    accepted = fem_state()
    proposal = next(p for p in construct_action_proposals(accepted.competition.hazard_states, correlation_interval_s=0.0) if p.action_type == "one_arm")
    result = execute_topology_trial(
        accepted, proposal, (arm(proposal.member_candidate_ids[0]),),
        apply_trial_geometry=lambda state, arms: apply_sharp_wake_trial_geometry(state, arms, kill_radius_m=0.2),
        equilibrate_fixed_load=equilibrate_to(7.0),
    )
    assert result.accepted is True
    assert result.energy_release_J_per_m == pytest.approx(3.0)
    assert result.state.crack_network.branch(ROOT_BRANCH_ID).tip == (1.5, 0.0)
    assert result.state.competition.consumed_event_ids == proposal.member_event_ids
    assert result.state.event_counters["topology_actions"] == 5


def test_one_arm_geometry_is_exact_sharp_wake_backend_parity():
    accepted = fem_state()
    proposal = next(p for p in construct_action_proposals(accepted.competition.hazard_states, correlation_interval_s=0.0) if p.action_type == "one_arm")
    trial_arm = arm(proposal.member_candidate_ids[0])
    direct = SharpWakeBackend().advance(
        mesh=accepted.mesh, boundary=accepted.boundary, damage=accepted.damage,
        displacement=accepted.displacement, p0=np.asarray(trial_arm.start_xy_m),
        p1=np.asarray(trial_arm.end_xy_m), direction=np.array([1.0, 0.0]),
        front_id=0, kill_r=0.2,
    )
    result = execute_topology_trial(
        accepted, proposal, (trial_arm,),
        apply_trial_geometry=lambda state, arms: apply_sharp_wake_trial_geometry(state, arms, kill_radius_m=0.2),
        equilibrate_fixed_load=equilibrate_to(7.0),
    )
    assert result.accepted
    np.testing.assert_array_equal(result.state.damage, direct.damage)
    np.testing.assert_array_equal(result.state.displacement, direct.displacement + 0.25)
    assert result.state.crack_network.total_physical_crack_length_m == pytest.approx(
        accepted.crack_network.total_physical_crack_length_m + direct.moved
    )


def test_two_arm_joint_energy_and_conservative_unresolved_cluster():
    accepted = fem_state()
    proposal = next(p for p in construct_action_proposals(accepted.competition.hazard_states, correlation_interval_s=0.0) if p.action_type == "two_arm")
    trial_network, cluster = create_unresolved_branch_cluster(
        accepted.crack_network, parent_branch_id=ROOT_BRANCH_ID,
        candidate_ids=proposal.member_candidate_ids, event_index=5,
        shared_process_state={"tip_state": accepted.tip_process_state},
        conserved_ledgers={
            "retained": 7.0, "mobile": 2.0, "escaped": 1.0,
            "recovered": 0.5, "stored_energy": 4.0,
            "emission_work": 3.0, "unconsumed_action": 0.25,
        },
    )
    arms = tuple(
        arm(candidate_id, branch_id, end=(1.4, y), dissipation=1.25)
        for (candidate_id, branch_id), y in zip(
            zip(proposal.member_candidate_ids, cluster.arm_branch_ids), (-0.3, 0.3)
        )
    )

    def geometry(state, trial_arms):
        state = replace(
            state, crack_network=trial_network,
            junction_process_state={"cluster": cluster},
        )
        return apply_sharp_wake_trial_geometry(state, trial_arms, kill_radius_m=0.2)

    result = execute_topology_trial(
        accepted, proposal, arms, apply_trial_geometry=geometry,
        equilibrate_fixed_load=equilibrate_to(7.0),
    )
    assert result.accepted
    assert result.hazard_dissipation_J_per_m == pytest.approx(2.5)
    assert len(result.state.crack_network.active_tip_ids) == 2
    assert result.state.competition.consumed_event_ids == proposal.member_event_ids
    assert cluster.unresolved
    assert cluster.conserved_ledgers["retained"] == 7.0

    first, second = cluster.arm_branch_ids
    independently_advanced = extend_network_arm(
        result.state.crack_network,
        TopologyArm(
            candidate_id=proposal.member_candidate_ids[0], branch_id=first,
            start_xy_m=(1.4, -0.3), end_xy_m=(1.8, -0.3),
            event_reward_m=0.4, hazard_dissipation_J_per_m=1.0,
        ),
    )
    assert independently_advanced.branch(first).tip == (1.8, -0.3)
    assert independently_advanced.branch(second).tip == (1.4, 0.3)


def test_independent_handoff_rejects_nonconservative_partition():
    network, cluster = create_unresolved_branch_cluster(
        fem_state().crack_network, parent_branch_id=ROOT_BRANCH_ID,
        candidate_ids=("c1", "c2"), event_index=1,
        shared_process_state={}, conserved_ledgers={name: 2.0 for name in (
            "retained", "mobile", "escaped", "recovered", "stored_energy",
            "emission_work", "unconsumed_action",
        )},
    )
    assert len(network.active_tip_ids) == 2
    with pytest.raises(ValueError, match="nonconservative"):
        cluster.handoff((
            {name: 1.0 for name in cluster.conserved_ledgers},
            {name: 0.5 for name in cluster.conserved_ledgers},
        ))


def test_branch_birth_is_candidate_label_and_insertion_order_invariant():
    base = fem_state().crack_network
    kwargs = dict(
        parent_branch_id=ROOT_BRANCH_ID, event_index=9,
        shared_process_state={"shared": True},
        conserved_ledgers={name: 1.0 for name in (
            "retained", "mobile", "escaped", "recovered", "stored_energy",
            "emission_work", "unconsumed_action",
        )},
    )
    network_a, cluster_a = create_unresolved_branch_cluster(
        base, candidate_ids=("physical-b", "physical-a"), **kwargs
    )
    network_b, cluster_b = create_unresolved_branch_cluster(
        base, candidate_ids=("physical-a", "physical-b"), **kwargs
    )
    assert network_a == network_b
    assert cluster_a == cluster_b


@pytest.mark.parametrize("after_branch", [False, True])
def test_restart_before_and_after_branch_birth(tmp_path, after_branch):
    state = fem_state()
    if after_branch:
        proposal = next(p for p in construct_action_proposals(state.competition.hazard_states, correlation_interval_s=0.0) if p.action_type == "two_arm")
        network, cluster = create_unresolved_branch_cluster(
            state.crack_network, parent_branch_id=ROOT_BRANCH_ID,
            candidate_ids=proposal.member_candidate_ids, event_index=5,
            shared_process_state={"retained": 7.0},
            conserved_ledgers={name: 0.0 for name in (
                "retained", "mobile", "escaped", "recovered", "stored_energy",
                "emission_work", "unconsumed_action",
            )},
        )
        state = replace(state, crack_network=network, junction_process_state={"cluster": cluster})
    path = tmp_path / "checkpoint.json"
    manifest = write_checkpoint(state, path)
    restored = restore_checkpoint(path)
    assert restored.crack_network == state.crack_network
    assert restored.competition == state.competition
    assert restored.tip_process_state == state.tip_process_state
    assert restored.junction_process_state == state.junction_process_state
    assert restored.energy_ledgers == state.energy_ledgers
    assert restored.rng_state == state.rng_state
    np.testing.assert_array_equal(restored.damage, state.damage)
    assert manifest["active_tip_ids"] == list(state.crack_network.active_tip_ids)


def test_restart_preserves_locked_provider_identity_and_topology_fingerprint(tmp_path):
    state = fem_state()
    runtime = LiveTopologyRuntime(
        str(tmp_path / "cache"),
        routing=replace(
            LiveTopologyRuntime(str(tmp_path / "cache")).routing,
            active_mechanics_provider="v11_exact_crack_network_live_fem_v1",
            transition_step=3, transition_state_hash="accepted",
            topology_fingerprint="topology-sha",
        ),
    )
    path = tmp_path / "provider-checkpoint.json"
    write_checkpoint(state, path, provider_runtime=runtime)
    restored, restored_runtime = restore_checkpoint(path, with_provider_runtime=True)
    assert restored.crack_network == state.crack_network
    np.testing.assert_array_equal(restored.damage, state.damage)
    assert restored_runtime == runtime
    assert restored_runtime.routing.topology_fingerprint == "topology-sha"


def test_first_intersection_clips_and_deactivates_only_incoming_tip(tmp_path):
    base = fem_state().crack_network
    network, cluster = create_unresolved_branch_cluster(
        base, parent_branch_id=ROOT_BRANCH_ID, candidate_ids=("c1", "c2"),
        event_index=1, shared_process_state={},
        conserved_ledgers={name: 0.0 for name in (
            "retained", "mobile", "escaped", "recovered", "stored_energy",
            "emission_work", "unconsumed_action",
        )},
    )
    first, second = cluster.arm_branch_ids
    network = extend_network_arm(network, TopologyArm("c1", first, (1.0, 0.0), (2.0, 1.0), 2**0.5, 0.0))
    network = extend_network_arm(network, TopologyArm("c2", second, (1.0, 0.0), (1.5, 1.0), 5**0.5 / 2, 0.0))
    incoming = TopologyArm("c2", second, (1.5, 1.0), (2.5, 0.0), 2**0.5, 0.0)
    clipped, target = clip_arm_at_first_intersection(network, incoming)
    assert target == first
    assert clipped.end_xy_m == pytest.approx((1.75, 0.75))
    network = extend_network_arm(network, clipped)
    network = mark_coalesced(network, second, target)
    assert network.branch(second).status == "merged"
    assert network.branch(first).status == "active"
    assert network.branch(first).path[-1] == (2.0, 1.0)

    state = replace(fem_state(), crack_network=network)
    path = tmp_path / "merged.json"
    write_checkpoint(state, path)
    restored = restore_checkpoint(path)
    assert restored.crack_network.branch(second).status == "merged"
    assert restored.crack_network.active_tip_ids == (first,)
