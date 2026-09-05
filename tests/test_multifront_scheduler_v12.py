from types import SimpleNamespace

from arrhenius_fracture.general_multifront_v12 import (
    GlobalSchedulerState, form_correlated_topology_proposals,
    select_global_topology_proposal,
)


def action(action_id, candidates, events, times, ordinals):
    return SimpleNamespace(
        action_id=action_id,
        action_type="two_arm" if len(candidates) == 2 else "one_arm",
        member_candidate_ids=tuple(candidates),
        member_event_ids=tuple(events),
        completion_times_s=tuple(times),
        member_event_ordinals=tuple(ordinals),
    )


def front_proposals(front, owner, *, pair_time=(2.0, 2.0 + 0.5e-7)):
    left = action("left", ("a",), ("a#1",), (pair_time[0],), (1,))
    right = action("right", ("b",), ("b#1",), (pair_time[1],), (1,))
    pair = action("pair", ("a", "b"), ("a#1", "b#1"), pair_time, (1, 1))
    endpoints = {"a": (1e-6, -1e-7), "b": (1e-6, 1e-7)}
    return form_correlated_topology_proposals(
        front_id=front, owner_id=owner,
        action_proposals=(right, pair, left), end_points_by_candidate=endpoints,
    )


def test_near_coincident_same_tip_members_form_one_atomic_proposal():
    proposals = front_proposals("front:a", "owner:a")
    assert len(proposals) == 1
    assert proposals[0].action_type == "two_arm"
    assert proposals[0].member_event_ids == ("a#1", "b#1")
    assert proposals[0].completion_time_s == 2.0 + 0.5e-7


def test_earlier_remote_one_arm_beats_later_atomic_two_arm_for_both_policies():
    pair = front_proposals("front:a", "owner:a", pair_time=(3.0, 3.0 + 0.5e-7))[0]
    remote = form_correlated_topology_proposals(
        front_id="front:b", owner_id="owner:b",
        action_proposals=(action("remote", ("c",), ("c#1",), (2.0,), (1,)),),
        end_points_by_candidate={"c": (1e-6, 0.0)},
    )[0]
    scheduler = GlobalSchedulerState(global_hazard_seed=3621, competition_event_index=23)
    for policy in (
        "v11_correlated_proposal_compatibility", "v12_global_earliest_proposal",
    ):
        assert select_global_topology_proposal(
            (pair, remote), scheduler, scheduler_policy=policy,
        ) == remote


def test_earlier_atomic_two_arm_beats_remote_one_arm():
    pair = front_proposals("front:a", "owner:a")[0]
    remote = form_correlated_topology_proposals(
        front_id="front:b", owner_id="owner:b",
        action_proposals=(action("remote", ("c",), ("c#1",), (3.0,), (1,)),),
        end_points_by_candidate={"c": (1e-6, 0.0)},
    )[0]
    scheduler = GlobalSchedulerState()
    assert select_global_topology_proposal(
        (remote, pair), scheduler,
        scheduler_policy="v12_global_earliest_proposal",
    ) == pair


def test_global_selection_is_enumeration_and_dictionary_order_invariant():
    first = front_proposals("front:a", "owner:a")[0]
    second = front_proposals("front:b", "owner:b")[0]
    scheduler = GlobalSchedulerState(tie_tolerance_s=1e-12, global_hazard_seed=91)
    for policy in (
        "v11_correlated_proposal_compatibility", "v12_global_earliest_proposal",
    ):
        a = select_global_topology_proposal((first, second), scheduler, scheduler_policy=policy)
        b = select_global_topology_proposal((second, first), scheduler, scheduler_policy=policy)
        assert a == b
