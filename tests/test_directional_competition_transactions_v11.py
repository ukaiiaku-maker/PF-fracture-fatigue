from dataclasses import replace

import pytest

from arrhenius_fracture.directional_competition_v11 import (
    DirectionalCompetitionState,
    accept_reservation,
    commit_directional_interval,
    construct_action_proposals,
    events_are_correlated,
    preview_directional_interval,
    release_reservation,
    reserve_action,
    tungsten_cleavage_candidates,
)


def pending_state(*, second_time=1.05, multiple=False):
    candidates = tungsten_cleavage_candidates(theta_deg=30.0)
    state = DirectionalCompetitionState.initialize(candidates, global_hazard_seed=1720)
    hazards = []
    for index, hazard in enumerate(state.hazard_states):
        duration = (2.2 if multiple and index == 0 else 1.0)
        start = 0.0 if index == 0 else second_time - 1.0
        preview = preview_directional_interval(
            hazard, lambda_per_s=1.0, start_time_s=start, duration_s=duration
        )
        hazards.append(commit_directional_interval(hazard, preview))
    return replace(state, hazard_states=tuple(hazards))


@pytest.mark.parametrize(
    "difference, expected",
    [(0.099999, True), (0.1, True), (0.100001, False)],
)
def test_correlation_boundary(difference, expected):
    state = pending_state(second_time=1.0 + difference)
    first, second = state.pending_events[:2]
    assert events_are_correlated(first, second, correlation_interval_s=0.1) is expected


def test_proposals_include_all_one_arm_and_only_binary_correlated_actions():
    state = pending_state(multiple=True)
    proposals = construct_action_proposals(
        state.hazard_states, correlation_interval_s=0.1
    )
    assert sum(item.action_type == "one_arm" for item in proposals) == 3
    assert all(len(item.member_event_ids) <= 2 for item in proposals)
    assert any(item.action_type == "two_arm" for item in proposals)


def test_released_reservation_consumes_nothing_and_unlocks_events():
    state = pending_state()
    proposal = next(
        item for item in construct_action_proposals(
            state.hazard_states, correlation_interval_s=0.1
        ) if item.action_type == "two_arm"
    )
    reserved = reserve_action(state, proposal, event_rewards_m=(1e-6, 2e-6))
    released = release_reservation(reserved, proposal.action_id)
    assert released.pending_events == state.pending_events
    assert released.consumed_event_ids == ()
    assert released.reservations[0].status == "released"


def test_same_pending_event_cannot_be_reserved_twice():
    state = pending_state()
    proposals = construct_action_proposals(state.hazard_states, correlation_interval_s=0.1)
    one = next(item for item in proposals if item.action_type == "one_arm")
    overlapping = next(
        item for item in proposals
        if item.action_type == "two_arm" and one.member_event_ids[0] in item.member_event_ids
    )
    reserved = reserve_action(state, one, event_rewards_m=(1e-6,))
    with pytest.raises(ValueError, match="already reserved"):
        reserve_action(reserved, overlapping, event_rewards_m=(1e-6, 2e-6))


@pytest.mark.parametrize("action_type, expected_consumed", [("one_arm", 1), ("two_arm", 2)])
def test_accept_consumes_exact_declared_ordinals(action_type, expected_consumed):
    state = pending_state()
    proposal = next(
        item for item in construct_action_proposals(
            state.hazard_states, correlation_interval_s=0.1
        ) if item.action_type == action_type
    )
    reserved = reserve_action(
        state, proposal, event_rewards_m=(1e-6,) * len(proposal.member_event_ids)
    )
    accepted = accept_reservation(reserved, proposal.action_id)
    assert accepted.consumed_event_ids == tuple(sorted(proposal.member_event_ids))
    assert len(state.pending_events) - len(accepted.pending_events) == expected_consumed
    assert accepted.reservations[0].status == "accepted"
    assert accepted.competition_event_index == 1


def test_every_available_proposal_consumes_exactly_its_declared_members():
    state = pending_state(multiple=True)
    proposals = construct_action_proposals(
        state.hazard_states, correlation_interval_s=0.1
    )
    for proposal in proposals:
        reserved = reserve_action(
            state, proposal, event_rewards_m=(1e-6,) * len(proposal.member_event_ids)
        )
        accepted = accept_reservation(reserved, proposal.action_id)
        assert accepted.consumed_event_ids == tuple(sorted(proposal.member_event_ids))
        assert set(accepted.pending_events) == {
            event for event in state.pending_events
            if event.event_id not in proposal.member_event_ids
        }


def test_rejected_unreserved_action_changes_nothing():
    state = pending_state()
    snapshot = state
    assert state == snapshot
