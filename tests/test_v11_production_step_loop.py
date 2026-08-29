from dataclasses import replace

import pytest

from arrhenius_fracture.directional_competition_v11 import (
    DirectionalCompetitionState, DirectionalRate, accept_reservation, reserve_action,
    DirectionalHazardState,
    tungsten_cleavage_candidates,
)
from arrhenius_fracture.production_step_loop_v11 import (
    AcceptedStepContext, DirectionalStepRefinementRequired, advance_accepted_step,
)
from arrhenius_fracture.multi_tip_step_loop_v11 import advance_multi_tip_step
from arrhenius_fracture.topology_transaction_v11 import TopologyTrialResult
from tests.test_topology_transaction_v11 import fem_state


def _rates(state, _context, values=(1.0, 1.0)):
    return tuple(
        DirectionalRate(c.candidate_id, rate, 2.0, 2.0, 3.0, c.gamma_rel)
        for c, rate in zip(state.competition.candidates, values)
    )


def _result(base, proposal, *, accepted):
    if accepted:
        reserved = reserve_action(
            base.competition, proposal,
            event_rewards_m=(0.5,) * len(proposal.member_event_ids),
        )
        state = replace(
            base,
            competition=accept_reservation(reserved, proposal.action_id),
            event_counters={**base.event_counters, "trial": len(proposal.member_event_ids)},
        )
    else:
        state = base
    return TopologyTrialResult(
        accepted, state, proposal.action_id, 3.0 if accepted else 0.5,
        2.0, 1.0 if accepted else -1.5,
        None if accepted else "insufficient_whole_topology_energy_release",
    )


def test_no_event_interval_updates_shared_state_once_without_trials():
    base = fem_state(DirectionalCompetitionState.initialize(
        tungsten_cleavage_candidates(theta_deg=45), global_hazard_seed=3621
    ))
    calls = []
    result = advance_accepted_step(
        base, AcceptedStepContext(1, 0.0, 0.25, "s0"), correlation_interval_s=0.1,
        solve_accepted=lambda state, context: state,
        evaluate_directional_rates=lambda state, context: _rates(state, context, (1.0, 1.0)),
        trial_action=lambda *_: pytest.fail("no trial is allowed"),
        update_shared_state_once=lambda state, context, proposal: calls.append(proposal) or state,
    )
    assert result.proposals == ()
    assert result.shared_state_update_count == 1
    assert calls == [None]


def test_a1_a2_a12_are_sibling_trials_and_admissible_a12_wins():
    base = fem_state(DirectionalCompetitionState.initialize(
        tungsten_cleavage_candidates(theta_deg=45), global_hazard_seed=3621
    ))
    base = replace(base, competition=replace(base.competition, hazard_states=tuple(
        DirectionalHazardState(candidate.candidate_id)
        for candidate in base.competition.candidates
    )))
    trial_bases = []
    updates = []

    def trial(state, proposal):
        trial_bases.append(state)
        return _result(state, proposal, accepted=True)

    result = advance_accepted_step(
        base, AcceptedStepContext(2, 0.0, 1.0, "s1"), correlation_interval_s=0.0,
        solve_accepted=lambda state, context: state,
        evaluate_directional_rates=_rates, trial_action=trial,
        update_shared_state_once=lambda state, context, proposal: updates.append(proposal) or state,
    )
    assert len(result.proposals) == 3
    assert all(item is trial_bases[0] for item in trial_bases)
    selected = next(item for item in result.trials if item.selected)
    assert selected.proposal.action_type == "two_arm"
    assert result.state.competition.consumed_event_ids == selected.proposal.member_event_ids
    assert len(updates) == 1


def test_rejected_a12_rolls_back_then_earliest_admissible_arm_commits():
    base = fem_state(DirectionalCompetitionState.initialize(
        tungsten_cleavage_candidates(theta_deg=45), global_hazard_seed=3621
    ))

    def trial(state, proposal):
        return _result(state, proposal, accepted=proposal.action_type == "one_arm")

    result = advance_accepted_step(
        base, AcceptedStepContext(2, 0.0, 1.0, "s2"), correlation_interval_s=0.0,
        solve_accepted=lambda state, context: state,
        evaluate_directional_rates=_rates, trial_action=trial,
        update_shared_state_once=lambda state, context, proposal: state,
    )
    selected = next(item for item in result.trials if item.selected)
    assert selected.proposal.action_type == "one_arm"
    assert len(result.state.competition.pending_events) == 1
    assert len(result.state.competition.consumed_event_ids) == 1


def test_rejected_trial_returning_modified_state_fails_closed():
    base = fem_state(DirectionalCompetitionState.initialize(
        tungsten_cleavage_candidates(theta_deg=45), global_hazard_seed=3621
    ))
    with pytest.raises(RuntimeError, match="rejected topology trial mutated"):
        advance_accepted_step(
            base, AcceptedStepContext(2, 0.0, 1.0, "s3"), correlation_interval_s=0.0,
            solve_accepted=lambda state, context: state,
            evaluate_directional_rates=_rates,
            trial_action=lambda state, proposal: TopologyTrialResult(
                False, state.isolated_copy(), proposal.action_id, 0.0, 1.0, -1.0, "veto"
            ),
            update_shared_state_once=lambda state, context, proposal: state,
        )


def test_adaptive_gate_rejects_unresolved_multi_event_interval_before_mutation():
    base = fem_state(DirectionalCompetitionState.initialize(
        tungsten_cleavage_candidates(theta_deg=45), global_hazard_seed=3621
    ))
    with pytest.raises(DirectionalStepRefinementRequired) as caught:
        advance_accepted_step(
            base, AcceptedStepContext(1, 0.0, 1.0, "adaptive"),
            correlation_interval_s=0.0,
            solve_accepted=lambda state, context: state,
            evaluate_directional_rates=lambda state, context: _rates(state, context, (2.0, 3.0)),
            trial_action=lambda *_: pytest.fail("refinement must precede topology trials"),
            update_shared_state_once=lambda *_: pytest.fail("refinement must precede state update"),
            maximum_directional_action_increment=0.15,
        )
    assert caught.value.predicted_increment == pytest.approx(3.0)
    assert base.competition.pending_events == ()


def test_pending_multi_tip_event_is_revalidated_against_current_signed_J():
    competition = DirectionalCompetitionState.initialize(
        tungsten_cleavage_candidates(theta_deg=45), global_hazard_seed=3621
    )
    base = fem_state(competition)
    tip = base.crack_network.active_tip_ids[0]
    first = advance_multi_tip_step(
        base, {tip: competition}, AcceptedStepContext(2, 0.0, 1.0, "current-network"),
        correlation_interval_s=0.0,
        solve_accepted=lambda state, context: state,
        evaluate_rates=lambda state, context: {tip: _rates(state, context)},
        trial_action=lambda state, tip_id, proposal: _result(state, proposal, accepted=False),
        update_process_states=lambda state, context, selected_tip, proposal: state,
    )
    assert first.competitions[tip].pending_events
    trial_calls = []
    negative_current_rates = tuple(
        DirectionalRate(candidate.candidate_id, 0.0, -1.0, 0.0, 0.0, candidate.gamma_rel)
        for candidate in competition.candidates
    )
    result = advance_multi_tip_step(
        first.state, first.competitions,
        AcceptedStepContext(3, 1.0, 0.1, "changed-network"),
        correlation_interval_s=0.0,
        solve_accepted=lambda state, context: state,
        evaluate_rates=lambda state, context: {tip: negative_current_rates},
        trial_action=lambda *args: trial_calls.append(args) or pytest.fail(
            "nonpositive current signed J must veto before a topology trial"
        ),
        update_process_states=lambda state, context, selected_tip, proposal: state,
    )
    assert trial_calls == []
    assert result.selected_proposal is None
    assert result.trials
    assert all(
        item.diagnostic.result.rejection_reason == "current_signed_directional_J_nonpositive"
        for item in result.trials
    )
    assert result.competitions[tip].pending_events
