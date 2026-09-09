from dataclasses import replace
import hashlib
import math
import random

import pytest

from arrhenius_fracture.crack_network_v11 import CrackNetworkState
from arrhenius_fracture.general_multifront_v12 import (
    CouplingEvidence, FrontCandidateObservation, FrontRuntimeState,
    MultiFrontRuntimeState, POLICY_BOUND_FRONT_LIMIT,
    POLICY_BOUND_TRANSACTION_LIMIT, ProcessEngineState, ResourcePolicy,
    TopologyProposal, advance_accepted_interval, commit_selected_proposal,
    kernel_coordinate_for_owner, recompute_process_region_connectivity,
    reject_proposal,
)


CANDIDATES = ("cleave:010", "cleave:100")


def initial_state(
    *, branching_mode="mechanistic", front_limit=None, transaction_limit=None,
    active_ledgers=None, shared_region_branching="experimental",
):
    network = CrackNetworkState.one_tip(
        ((0.0, 0.0),), initial_orientation_rad=0.0,
    )
    front_id = network.active_tip_ids[0]
    runtime = FrontRuntimeState(
        front_id=front_id, competition_state={"action": 0.0},
        candidate_ids=CANDIDATES, lineage_rng_state={"seed": 7},
    )
    engine = ProcessEngineState(
        engine_id="engine:root", source_state_id="source:qualified-v5.3",
        active_ledgers=active_ledgers or {"mobile": 3.0, "retained": 2.0},
        wake_ledgers={"mobile": 1.0},
        signed_system_ledgers={"positive": 4.0, "negative": 2.0},
    )
    return MultiFrontRuntimeState.one_front(
        network, runtime, engine,
        resource_policy=ResourcePolicy(
            branching_mode=branching_mode,
            front_resource_limit=front_limit,
            branch_transaction_limit=transaction_limit,
            shared_region_branching=shared_region_branching,
        ),
        accepted_state_id="accepted:0", stress_field_state_id="stress:0",
    )


def test_process_engine_accepts_finite_signed_system_totals_only():
    accepted = replace(
        next(iter(initial_state().process_engines.values())),
        signed_system_ledgers={"net_mobile": -2.5},
    )
    assert accepted.signed_system_ledgers == {"net_mobile": -2.5}
    with pytest.raises(ValueError, match="signed_ledger"):
        replace(
            accepted,
            signed_system_ledgers={"net_mobile": float("nan")},
        )


def test_binary_birth_reuses_exact_v11_branch_and_cluster_id_contract():
    state = initial_state(front_limit=2)
    parent = state.active_front_ids[0]
    item = proposal(state, parent, "two_arm")
    result = commit_selected_proposal(state, item)
    expected_children = tuple(sorted(
        "b" + hashlib.sha256(
            f"{parent}|{candidate}|1".encode()
        ).hexdigest()[:15]
        for candidate in item.candidate_ids
    ))
    expected_junction = "j" + hashlib.sha256(
        f"{parent}|1|{'|'.join(sorted(item.candidate_ids))}".encode()
    ).hexdigest()[:15]
    assert result.active_front_ids == expected_children
    assert tuple(result.junctions) == (expected_junction,)
    assert result.crack_network.geometry_generation == (
        state.crack_network.geometry_generation + 2
    )
    for child_id in expected_children:
        child = result.crack_network.branch(child_id)
        assert child.local_state["cluster_unresolved"] is True
        assert len(child.local_state["committed_edges"]) == 1


def proposal(state, front_id, action_type="two_arm", *, completion=1.0, sequence=0):
    owner = state.owner_by_front[front_id]
    tip = state.crack_network.branch(front_id).tip
    if action_type == "two_arm":
        candidates = CANDIDATES
        ends = (
            (tip[0] + 1.0e-6, tip[1] - (sequence + 1) * 1.0e-9),
            (tip[0] + 1.2e-6, tip[1] + (sequence + 1) * 1.0e-9),
        )
    elif action_type == "one_arm":
        candidates = (CANDIDATES[0],)
        ends = ((tip[0] + 1.1e-6, tip[1]),)
    else:
        candidates, ends = (), ()
    return TopologyProposal(
        proposal_id=f"p:{sequence}:{front_id}:{action_type}",
        action_type=action_type, front_id=front_id, owner_id=owner,
        candidate_ids=candidates, completion_time_s=completion,
        end_points_m=ends,
    )


def branch_to_count(count):
    state = initial_state()
    for index in range(count - 1):
        front_id = state.active_front_ids[index % len(state.active_front_ids)]
        fronts = dict(state.front_runtimes)
        fronts[front_id] = fronts[front_id].activate_complete_competition()
        state = replace(state, front_runtimes=fronts)
        state = commit_selected_proposal(
            state, proposal(state, front_id, sequence=index),
        )
    return state


def observations(state, *, reverse=False):
    rows = []
    for front_index, front_id in enumerate(state.active_front_ids):
        owner = state.owner_by_front[front_id]
        tip = state.crack_network.branch(front_id).tip
        active = state.front_runtimes[front_id].mechanically_active_candidate_ids
        for candidate_index, candidate in enumerate(active):
            rows.append(FrontCandidateObservation(
                accepted_state_id=state.accepted_state_id,
                stress_field_state_id=state.stress_field_state_id,
                front_id=front_id, owner_id=owner, candidate_id=candidate,
                tip_coordinates_m=tip, signed_local_J_J_per_m2=10.0,
                marginal_G_J_per_m2=11.0, kinetic_J_used_J_per_m2=10.0,
                directional_K_MPa_sqrt_m=20.0 + front_index + candidate_index / 10,
                directional_rate_per_s=0.1, tensor=(1.0, 2.0, 3.0),
                tensor_reliability="qualified",
                controlling_scalar_K_tip_id=front_id, tensor_probe_tip_id=front_id,
            ))
    return tuple(reversed(rows)) if reverse else tuple(rows)


@pytest.mark.parametrize("count", (1, 2, 3, 4, 8, 16, 32))
def test_repeated_binary_transactions_support_arbitrary_exercised_front_counts(count):
    state = branch_to_count(count)
    assert len(state.active_front_ids) == count
    assert state.cumulative_branch_births == count - 1
    assert len(state.junctions) == count - 1
    assert len(state.process_regions) == 1
    assert len(state.process_engines) == 1
    state.validate()


def test_recursive_birth_replaces_one_member_and_retains_one_engine():
    two = branch_to_count(2)
    owner = next(iter(two.process_regions))
    engine_id = two.process_regions[owner].process_engine_id
    parent = two.active_front_ids[0]
    fronts = dict(two.front_runtimes)
    fronts[parent] = fronts[parent].activate_complete_competition()
    two = replace(two, front_runtimes=fronts)
    three = commit_selected_proposal(two, proposal(two, parent, sequence=2))
    assert len(three.active_front_ids) == 3
    assert parent not in three.owner_by_front
    assert set(three.owner_by_front.values()) == {owner}
    assert three.process_regions[owner].process_engine_id == engine_id
    assert len(three.process_regions[owner].member_front_ids) == 3


def test_binary_candidate_order_is_canonical_without_swapping_geometry():
    state = initial_state()
    front = state.active_front_ids[0]
    normal = proposal(state, front)
    reversed_input = TopologyProposal(
        proposal_id=normal.proposal_id, action_type="two_arm", front_id=front,
        owner_id=normal.owner_id, candidate_ids=tuple(reversed(normal.candidate_ids)),
        completion_time_s=normal.completion_time_s,
        end_points_m=tuple(reversed(normal.end_points_m)),
    )
    assert normal == reversed_input
    assert commit_selected_proposal(state, normal).topology_fingerprint == commit_selected_proposal(
        state, reversed_input
    ).topology_fingerprint


def test_global_scheduler_selects_earlier_one_arm_over_later_two_arm():
    state = branch_to_count(2)
    first, second = state.active_front_ids
    early = proposal(state, first, "one_arm", completion=2.0, sequence=1)
    late = proposal(state, second, "two_arm", completion=3.0, sequence=2)
    result = advance_accepted_interval(
        state, observations(state), (late, early), duration_s=0.5,
    )
    assert result.selected_proposal == early


def test_numerical_tie_is_identity_deterministic_and_order_invariant():
    state = branch_to_count(2)
    left, right = state.active_front_ids
    values = (
        proposal(state, left, "one_arm", completion=2.0, sequence=1),
        proposal(state, right, "two_arm", completion=2.0 + 0.5e-12, sequence=2),
    )
    a = advance_accepted_interval(state, observations(state), values, duration_s=0.5)
    b = advance_accepted_interval(state, observations(state, reverse=True), tuple(reversed(values)), duration_s=0.5)
    assert a.selected_proposal == b.selected_proposal


def test_one_update_per_owner_not_per_front_and_zero_renewal_during_time_update():
    state = branch_to_count(8)
    before = next(iter(state.process_engines.values()))
    result = advance_accepted_interval(state, observations(state), (), duration_s=1.0)
    after = next(iter(result.state.process_engines.values()))
    assert after.update_count == before.update_count + 1
    assert after.event_renewal_count == before.event_renewal_count
    assert all(item.interval_count == 1 for item in result.state.front_runtimes.values())


def test_incomplete_or_mixed_tip_observation_fails_closed():
    state = branch_to_count(2)
    with pytest.raises(ValueError, match="cover every"):
        advance_accepted_interval(state, observations(state)[:-1], (), duration_s=1.0)
    row = observations(state)[0]
    with pytest.raises(ValueError, match="same front"):
        replace(row, tensor_probe_tip_id=state.active_front_ids[1])


def test_resource_limit_terminates_without_suppressing_or_mutating_completed_birth():
    state = initial_state(front_limit=1)
    front = state.active_front_ids[0]
    stopped = commit_selected_proposal(state, proposal(state, front))
    assert stopped.termination_reason == POLICY_BOUND_FRONT_LIMIT
    assert stopped.policy_bound
    assert stopped.topology_fingerprint == state.topology_fingerprint
    assert stopped.registry_fingerprint == state.registry_fingerprint
    assert not stopped.transaction_records


def test_branch_transaction_limit_is_independent_of_active_front_count():
    state = initial_state(transaction_limit=1)
    state = commit_selected_proposal(state, proposal(state, state.active_front_ids[0]))
    stopped = commit_selected_proposal(state, proposal(state, state.active_front_ids[0], sequence=2))
    assert stopped.termination_reason == POLICY_BOUND_TRANSACTION_LIMIT
    assert len(stopped.active_front_ids) == 2
    assert stopped.cumulative_branch_births == 1


def test_branching_disabled_control_fails_before_topology_mutation():
    state = initial_state(branching_mode="disabled", front_limit=1)
    with pytest.raises(ValueError, match="disabled"):
        commit_selected_proposal(state, proposal(state, state.active_front_ids[0]))


def test_process_region_split_archives_once_and_fresh_renews_each_component():
    state = branch_to_count(2)
    junction_id = next(iter(state.junctions))
    before_ledgers = state.total_conserved_ledgers()
    before_signed = state.total_signed_system_ledgers()
    split = recompute_process_region_connectivity(state, {
        junction_id: CouplingEvidence(
            junction_id=junction_id, branch_handoff_length_m=5e-6,
            process_zone_length_m=2e-6, local_contour_radius_m=1e-6,
            actual_tip_separation_m=10e-6,
            minimum_post_junction_path_length_m=8e-6,
            contours_overlap=False,
        )
    })
    assert len(split.process_regions) == 2
    assert len(split.process_engines) == 2
    assert len(split.reservoirs) == 1
    assert split.total_conserved_ledgers() == before_ledgers
    assert split.total_signed_system_ledgers() == before_signed
    assert all(not engine.active_ledgers and not engine.wake_ledgers for engine in split.process_engines.values())
    assert len({region.process_engine_id for region in split.process_regions.values()}) == 2
    assert all(len(region.member_front_ids) == 1 for region in split.process_regions.values())
    reservoir_id = next(iter(split.reservoirs))
    assert all(region.source_reservoir_id == reservoir_id for region in split.process_regions.values())
    assert split.transaction_records[-1].action_type == "process_region_partition"


def test_every_independent_owner_updates_once_after_partition():
    state = branch_to_count(2)
    junction_id = next(iter(state.junctions))
    state = recompute_process_region_connectivity(state, {
        junction_id: CouplingEvidence(junction_id, 5e-6, 2e-6, 1e-6, 10e-6, 8e-6, False)
    })
    before = {key: item.update_count for key, item in state.process_engines.items()}
    result = advance_accepted_interval(state, observations(state), (), duration_s=1.0)
    assert all(
        result.state.process_engines[key].update_count == count + 1
        for key, count in before.items()
    )


def test_rejected_transaction_returns_identical_accepted_state_object():
    state = initial_state()
    item = proposal(state, state.active_front_ids[0], "one_arm")
    assert reject_proposal(state, item) is state


def test_nested_junction_graph_supports_asymmetric_partial_handoff():
    state = branch_to_count(3)
    junction_ids = sorted(state.junctions, key=lambda item: state.junctions[item].birth_transaction_id)
    first, second = junction_ids
    evidence = {
        first: CouplingEvidence(first, 5e-6, 2e-6, 1e-6, 20e-6, 8e-6, False),
        second: CouplingEvidence(second, 5e-6, 2e-6, 1e-6, 1e-6, 1e-6, True),
    }
    partial = recompute_process_region_connectivity(state, evidence)
    assert sorted(len(region.member_front_ids) for region in partial.process_regions.values()) == [1, 2]
    coupled_owner = next(
        owner for owner, region in partial.process_regions.items()
        if len(region.member_front_ids) == 2
    )
    assert partial.process_regions[coupled_owner].unresolved_junction_ids == {second}
    complete = recompute_process_region_connectivity(partial, {
        second: CouplingEvidence(second, 5e-6, 2e-6, 1e-6, 20e-6, 8e-6, False)
    })
    assert len(complete.process_regions) == 3
    assert all(len(region.member_front_ids) == 1 for region in complete.process_regions.values())
    # Partial handoff retains the historical engine with the unresolved pair;
    # only the later complete resolution archives it.
    assert len(partial.reservoirs) == 0
    assert len(complete.reservoirs) == 1


def test_sequential_asymmetric_detachment_resolves_orphaned_old_junction():
    state = branch_to_count(2)
    junction_id = next(iter(state.junctions))
    first, second = state.active_front_ids
    partial = recompute_process_region_connectivity(state, {
        junction_id: CouplingEvidence(
            junction_id, 5e-6, 2e-6, 1e-6, 1e-6, 8e-6, True,
            (first,),
        ),
    })
    retained_owner = partial.owner_by_front[second]
    assert partial.process_regions[retained_owner].unresolved_junction_ids == {
        junction_id
    }
    complete = recompute_process_region_connectivity(partial, {
        junction_id: CouplingEvidence(
            junction_id, 5e-6, 2e-6, 1e-6, 1e-6, 8e-6, True,
            (second,),
        ),
    })
    assert complete.junctions[junction_id].status == "resolved"
    assert all(
        junction_id not in region.unresolved_junction_ids
        for region in complete.process_regions.values()
    )


def test_local_j_never_enters_connectivity_evidence_or_blocks_split():
    assert "local_J" not in CouplingEvidence.__dataclass_fields__


def test_owner_local_kernel_coordinates_are_independent_after_split():
    state = branch_to_count(2)
    junction_id = next(iter(state.junctions))
    state = recompute_process_region_connectivity(state, {
        junction_id: CouplingEvidence(junction_id, 5e-6, 2e-6, 1e-6, 10e-6, 8e-6, False)
    })
    first, second = state.active_front_ids
    second_owner = state.owner_by_front[second]
    before_second = kernel_coordinate_for_owner(state, second_owner)
    state = commit_selected_proposal(state, proposal(state, first, "one_arm", sequence=7))
    assert kernel_coordinate_for_owner(state, second_owner) == before_second
    assert kernel_coordinate_for_owner(state, state.owner_by_front[first]) > 0.0


def test_owner_local_kernel_domain_fails_closed_per_owner():
    state = initial_state()
    owner = next(iter(state.process_regions))
    region = state.process_regions[owner]
    engine = state.process_engines[region.process_engine_id]
    engines = {engine.engine_id: replace(engine, local_process_coordinate_m=746e-6)}
    regions = {owner: replace(region, cumulative_process_advance_m=746e-6)}
    state = replace(state, process_engines=engines, process_regions=regions)
    with pytest.raises(ValueError, match="owner_local"):
        kernel_coordinate_for_owner(state, owner)


def test_coalescence_and_retirement_counts_are_not_inferred_from_births():
    state = branch_to_count(4)
    incoming, target, retired = state.active_front_ids[:3]
    merge = TopologyProposal(
        proposal_id="merge", action_type="coalescence", front_id=incoming,
        owner_id=state.owner_by_front[incoming], candidate_ids=(),
        completion_time_s=1.0, target_front_id=target,
    )
    state = commit_selected_proposal(state, merge)
    retire = TopologyProposal(
        proposal_id="retire", action_type="retirement", front_id=retired,
        owner_id=state.owner_by_front[retired], candidate_ids=(), completion_time_s=2.0,
    )
    state = commit_selected_proposal(state, retire)
    assert len(state.active_front_ids) == 2
    assert state.cumulative_branch_births == 3
    assert state.cumulative_coalescences == 1
    assert state.cumulative_retirements == 1


@pytest.mark.parametrize("seed", range(12))
def test_deterministic_property_sequences_preserve_registry_invariants(seed):
    rng = random.Random(seed)
    state = initial_state()
    for step in range(40):
        active = list(state.active_front_ids)
        if len(active) < 2 or rng.random() < 0.65:
            front = rng.choice(active)
            action = "two_arm" if len(active) < 12 and rng.random() < 0.5 else "one_arm"
            state = commit_selected_proposal(state, proposal(state, front, action, sequence=step))
        elif len(active) > 1:
            front = rng.choice(active)
            action = "retirement" if rng.random() < 0.5 else "coalescence"
            target = next((item for item in active if item != front), None)
            item = TopologyProposal(
                proposal_id=f"property:{seed}:{step}", action_type=action,
                front_id=front, owner_id=state.owner_by_front[front], candidate_ids=(),
                completion_time_s=float(step), target_front_id=(target if action == "coalescence" else None),
            )
            state = commit_selected_proposal(state, item)
        state.validate()
        assert len(state.front_runtimes) == len(state.active_front_ids)
        assert len(state.owner_by_front) == len(state.active_front_ids)
        assert len(state.process_engines) == len(state.process_regions)
        assert len({region.process_engine_id for region in state.process_regions.values()}) == len(state.process_regions)
