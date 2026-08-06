import random
from dataclasses import replace

from arrhenius_fracture.directional_competition_v11 import (
    CleavageCandidate,
    DirectionalCompetitionState,
    canonical_candidate_inventory,
    commit_directional_interval,
    competition_state_to_json,
    construct_action_proposals,
    preview_directional_interval,
    select_temporal_or_degenerate_proposal,
    tungsten_cleavage_candidates,
)


def test_equivalent_physical_identity_is_stable_and_variants_do_not_collide():
    first = CleavageCandidate.create(
        plane_family="cleavage", plane_variant="(100)",
        direction_xy=(2.0, -0.0), normal_xy=(0.0, 3.0), gamma_rel=1.0,
        orientation_convention="bcc:theta=0",
    )
    equivalent = CleavageCandidate.create(
        plane_family="cleavage", plane_variant="(100)",
        direction_xy=(1.0, 0.0), normal_xy=(-0.0, 1.0), gamma_rel=1.0,
        orientation_convention="bcc:theta=0",
    )
    distinct = CleavageCandidate.create(
        plane_family="cleavage", plane_variant="(010)",
        direction_xy=(1.0, 0.0), normal_xy=(0.0, 1.0), gamma_rel=1.0,
        orientation_convention="bcc:theta=0",
    )
    assert first.candidate_id == equivalent.candidate_id
    assert first.candidate_id != distinct.candidate_id


def _result(candidates):
    state = DirectionalCompetitionState.initialize(candidates, global_hazard_seed=99)
    hazards = []
    rates = {}
    for index, hazard in enumerate(state.hazard_states):
        rate = 1.0 + index
        rates[hazard.candidate_id] = rate
        preview = preview_directional_interval(
            hazard, lambda_per_s=rate, start_time_s=10.0, duration_s=2.1
        )
        hazards.append(commit_directional_interval(hazard, preview))
    state = replace(state, hazard_states=tuple(hazards))
    proposals = construct_action_proposals(state.hazard_states, correlation_interval_s=1.0)
    chosen = select_temporal_or_degenerate_proposal(
        proposals, global_hazard_seed=99, competition_event_index=4
    )
    return competition_state_to_json(state), rates, proposals, chosen.action_id


def test_candidate_permutation_does_not_change_state_proposals_or_tie_result():
    candidates = list(tungsten_cleavage_candidates(theta_deg=30.0, include_110=True))
    expected = _result(candidates)
    for seed in range(20):
        shuffled = candidates[:]
        random.Random(seed).shuffle(shuffled)
        assert _result(shuffled) == expected


def test_temporal_priority_is_seed_independent_but_degenerate_choice_is_repeatable():
    candidates = tungsten_cleavage_candidates(theta_deg=30.0)
    state = DirectionalCompetitionState.initialize(candidates, global_hazard_seed=1)
    hazards = []
    for hazard in state.hazard_states:
        preview = preview_directional_interval(
            hazard, lambda_per_s=1.0, start_time_s=0.0, duration_s=1.0
        )
        hazards.append(commit_directional_interval(hazard, preview))
    proposals = construct_action_proposals(hazards, correlation_interval_s=0.0)
    first = select_temporal_or_degenerate_proposal(
        proposals, global_hazard_seed=1720, competition_event_index=3
    )
    repeat = select_temporal_or_degenerate_proposal(
        tuple(reversed(proposals)), global_hazard_seed=1720, competition_event_index=3
    )
    assert first == repeat

    degenerate_choices = {
        select_temporal_or_degenerate_proposal(
            proposals, global_hazard_seed=seed, competition_event_index=3
        ).action_id
        for seed in range(64)
    }
    assert len(degenerate_choices) > 1

    earlier = replace(proposals[0], completion_times_s=(0.5,) * len(proposals[0].completion_times_s))
    later = replace(proposals[1], completion_times_s=(1.0,) * len(proposals[1].completion_times_s))
    for seed in (1, 2, 999):
        assert select_temporal_or_degenerate_proposal(
            (later, earlier), global_hazard_seed=seed, competition_event_index=3
        ) == earlier
