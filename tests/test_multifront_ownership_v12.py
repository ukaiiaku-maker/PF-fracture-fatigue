from dataclasses import replace

import pytest

from arrhenius_fracture.crack_network_v11 import CrackNetworkState
from arrhenius_fracture.general_multifront_v12 import (
    CouplingEvidence, FrontCandidateObservation, FrontRuntimeState,
    MultiFrontRuntimeState, PARENT_PROCESS_ZONE_UNRESOLVED, ProcessEngineState,
    ResourcePolicy, TopologyProposal, advance_accepted_interval,
    commit_selected_proposal, recompute_process_region_connectivity,
)


CANDIDATES = ("cleave:010", "cleave:100")


def initial(*, experimental=False):
    network = CrackNetworkState.one_tip(((0.0, 0.0),), initial_orientation_rad=0.0)
    front = network.active_tip_ids[0]
    runtime = FrontRuntimeState(
        front, {"action": 0.25, "threshold": 1.5, "ordinal": 3},
        CANDIDATES, {"seed": 91, "state": [1, 2, 3]},
    )
    engine = ProcessEngineState(
        "engine:historical", "family:v5.3",
        {"mobile": 3.0, "retained": 2.0}, {"retained": 7.0},
        {"positive": 9.0, "negative": 1.0},
        update_count=11, event_renewal_count=4,
        local_process_coordinate_m=21e-6,
        opaque_state_fingerprint="historical-complete",
        mutable_state={"B": 0.4, "N_em": 8.0}, rng_state={"seed": 201},
    )
    return MultiFrontRuntimeState.one_front(
        network, runtime, engine,
        resource_policy=ResourcePolicy(
            "mechanistic", None, None,
            "experimental" if experimental else "forbid",
        ), accepted_state_id="accepted", stress_field_state_id="stress",
    )


def proposal(state, front, action="two_arm", index=0, completion=1.0):
    tip = state.crack_network.branch(front).tip
    if action == "two_arm":
        candidates = CANDIDATES
        ends = ((tip[0] + 1e-6, tip[1] - 1e-7), (tip[0] + 1e-6, tip[1] + 1e-7))
    elif action == "one_arm":
        candidates, ends = (CANDIDATES[0],), ((tip[0] + 1e-6, tip[1]),)
    else:
        candidates, ends = (), ()
    return TopologyProposal(
        f"p:{index}:{front}:{action}", action, front,
        state.owner_by_front[front], candidates, completion, ends,
    )


def branch(state, front=None, index=0):
    return commit_selected_proposal(
        state, proposal(state, front or state.active_front_ids[0], index=index),
    )


def evidence(state, junction, *, coupled, detached=()):
    return CouplingEvidence(
        junction, 5e-6, 2e-6, 1e-6,
        1e-6 if coupled else 20e-6,
        1e-6 if coupled else 8e-6,
        coupled, tuple(detached),
    )


def observations(state):
    rows = []
    for front_index, front in enumerate(state.active_front_ids):
        active = state.front_runtimes[front].mechanically_active_candidate_ids
        for candidate_index, candidate in enumerate(active):
            rows.append(FrontCandidateObservation(
                state.accepted_state_id, state.stress_field_state_id,
                front, state.owner_by_front[front], candidate,
                state.crack_network.branch(front).tip, 1.0, 1.0, 1.0,
                20.0 + front_index + candidate_index / 10, 1.0,
                (1.0, 2.0, 3.0), "qualified", front, front,
            ))
    return tuple(rows)


def test_production_policy_forbids_recursive_branching_until_owner_resolves():
    state = branch(initial())
    with pytest.raises(ValueError, match=PARENT_PROCESS_ZONE_UNRESOLVED):
        branch(state, state.active_front_ids[0], 2)
    capped = replace(state, resource_policy=replace(
        state.resource_policy, front_resource_limit=2,
    ))
    with pytest.raises(ValueError, match=PARENT_PROCESS_ZONE_UNRESOLVED):
        branch(capped, capped.active_front_ids[0], 3)
    assert branch(branch(initial(experimental=True)), index=2).cumulative_branch_births == 2


def test_two_shared_arms_asymmetric_handoff_retains_whole_historical_engine():
    state = branch(initial(experimental=True))
    junction = next(iter(state.junctions))
    old_owner = next(iter(state.process_regions))
    old_region = state.process_regions[old_owner]
    old_engine = state.process_engines[old_region.process_engine_id]
    detached = state.active_front_ids[0]
    result = recompute_process_region_connectivity(state, {
        junction: evidence(state, junction, coupled=True, detached=(detached,)),
    })
    retained = result.process_regions[old_owner]
    assert retained.member_front_ids == {state.active_front_ids[1]}
    assert retained.unresolved_junction_ids == {junction}
    assert result.process_engines[old_engine.engine_id] is old_engine
    assert retained.cumulative_process_advance_m == old_region.cumulative_process_advance_m
    assert result.owner_by_front[detached] != old_owner
    fresh = result.process_engines[
        result.process_regions[result.owner_by_front[detached]].process_engine_id
    ]
    assert fresh.active_ledgers == fresh.wake_ledgers == {}
    assert fresh.opaque_state_fingerprint == "fresh_independent_engine_after_handoff"
    assert not result.reservoirs
    assert result.total_conserved_ledgers() == state.total_conserved_ledgers()
    assert result.total_signed_system_ledgers() == state.total_signed_system_ledgers()


def test_three_front_partial_then_residual_last_arm_archives_only_at_completion():
    state = initial(experimental=True)
    state = branch(state, index=0)
    state = branch(state, state.active_front_ids[0], index=1)
    first, second = sorted(state.junctions, key=lambda key: state.junctions[key].birth_transaction_id)
    partial = recompute_process_region_connectivity(state, {
        first: evidence(state, first, coupled=False),
        second: evidence(state, second, coupled=True),
    })
    assert sorted(len(item.member_front_ids) for item in partial.process_regions.values()) == [1, 2]
    assert not partial.reservoirs
    residual_owner = next(
        key for key, value in partial.process_regions.items() if value.unresolved_junction_ids
    )
    historical_engine = partial.process_regions[residual_owner].process_engine_id
    complete = recompute_process_region_connectivity(partial, {
        second: evidence(partial, second, coupled=False),
    })
    assert len(complete.reservoirs) == 1
    assert next(iter(complete.reservoirs.values())).archived_engine.engine_id == historical_engine
    assert all(len(item.member_front_ids) == 1 for item in complete.process_regions.values())


@pytest.mark.parametrize("action,status", (("retirement", "terminated"), ("coalescence", "merged")))
def test_shared_arm_retirement_or_coalescence_preserves_residual_historical_owner(action, status):
    state = branch(initial(experimental=True))
    removed, target = state.active_front_ids
    old_owner = state.owner_by_front[removed]
    item = TopologyProposal(
        f"{action}:1", action, removed, old_owner, (), 2.0, (),
        target_front_id=(target if action == "coalescence" else None),
    )
    result = commit_selected_proposal(state, item)
    assert result.crack_network.branch(removed).status == status
    assert result.owner_by_front[target] == old_owner
    assert result.process_regions[old_owner].unresolved_junction_ids
    assert not result.reservoirs


def multi_owner_state():
    state = initial(experimental=True)
    for index in range(3):
        front = state.active_front_ids[index % len(state.active_front_ids)]
        fronts = dict(state.front_runtimes)
        fronts[front] = fronts[front].activate_complete_competition()
        state = replace(state, front_runtimes=fronts)
        state = branch(state, front, index=index)
    state = recompute_process_region_connectivity(state, {
        key: evidence(state, key, coupled=False) for key in state.junctions
    })
    state = replace(state, resource_policy=replace(
        state.resource_policy, shared_region_branching="forbid",
    ))
    independent = state.active_front_ids
    state = branch(state, independent[0], index=10)
    state = branch(state, independent[1], index=11)
    return state


def test_two_unresolved_clusters_coexist_with_unrelated_independent_tips():
    state = multi_owner_state()
    assert len(state.active_front_ids) == 6
    assert sum(bool(item.unresolved_junction_ids) for item in state.process_regions.values()) == 2
    assert sum(not item.unresolved_junction_ids for item in state.process_regions.values()) == 2
    assert len(state.process_regions) == len(state.process_engines) == 4


def test_multiowner_interval_updates_once_and_renews_only_selected_owner():
    state = multi_owner_state()
    selected_front = next(
        front for front in state.active_front_ids
        if not state.process_regions[state.owner_by_front[front]].unresolved_junction_ids
    )
    selected_owner = state.owner_by_front[selected_front]
    item = proposal(state, selected_front, "one_arm", 20, 2.0)
    before = {
        owner: state.process_engines[region.process_engine_id]
        for owner, region in state.process_regions.items()
    }
    interval = advance_accepted_interval(
        state, observations(state), (item,), duration_s=0.5,
    )
    assert interval.selected_proposal == item
    assert len(interval.controlling_observation_by_owner) == len(state.process_regions)
    assert all(row.interval_count == 1 for row in interval.state.front_runtimes.values())
    assert len(interval.state.transaction_records) == len(state.transaction_records)
    assert interval.state.total_conserved_ledgers() == state.total_conserved_ledgers()
    assert interval.state.total_signed_system_ledgers() == state.total_signed_system_ledgers()
    assert {
        key: value.lineage_rng_state for key, value in interval.state.front_runtimes.items()
    } == {key: value.lineage_rng_state for key, value in state.front_runtimes.items()}
    assert all(
        interval.state.process_engines[state.process_regions[owner].process_engine_id].update_count
        == engine.update_count + 1 for owner, engine in before.items()
    )
    committed = commit_selected_proposal(interval.state, item)
    assert len(committed.transaction_records) == len(state.transaction_records) + 1
    for owner, engine in before.items():
        after = committed.process_engines[committed.process_regions[owner].process_engine_id]
        expected = engine.event_renewal_count + (owner == selected_owner)
        assert after.event_renewal_count == expected
        expected_coordinate = engine.local_process_coordinate_m + (
            committed.transaction_records[-1].renewal_distance_m if owner == selected_owner else 0.0
        )
        assert after.local_process_coordinate_m == expected_coordinate
