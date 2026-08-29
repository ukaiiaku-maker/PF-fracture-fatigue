from dataclasses import replace

from arrhenius_fracture.branch_policy_v11 import BRANCH_CAP_VETO, MAX_BRANCH_BIRTHS, branch_birth_policy
from arrhenius_fracture.directional_competition_v11 import (
    DirectionalCompetitionState, DirectionalHazardState, commit_directional_interval,
    construct_action_proposals, preview_directional_interval,
    tungsten_cleavage_candidates,
)


def _proposals():
    state = DirectionalCompetitionState.initialize(tungsten_cleavage_candidates(theta_deg=45), global_hazard_seed=3621)
    state = replace(state, hazard_states=tuple(
        DirectionalHazardState(candidate.candidate_id) for candidate in state.candidates
    ))
    state = replace(state, hazard_states=tuple(commit_directional_interval(
        hazard, preview_directional_interval(hazard, lambda_per_s=1.0, start_time_s=0.0, duration_s=1.0)
    ) for hazard in state.hazard_states))
    return construct_action_proposals(state.hazard_states, correlation_interval_s=1e-6)


def test_branch_cap_vetoes_only_new_A12_and_keeps_one_arm_propagation():
    proposals = _proposals()
    two = next(item for item in proposals if item.action_type == "two_arm")
    one = next(item for item in proposals if item.action_type == "one_arm")
    assert branch_birth_policy(two, committed_branch_birth_count=MAX_BRANCH_BIRTHS - 1).permitted
    veto = branch_birth_policy(two, committed_branch_birth_count=MAX_BRANCH_BIRTHS)
    assert not veto.permitted and veto.veto_reason == BRANCH_CAP_VETO
    assert branch_birth_policy(one, committed_branch_birth_count=MAX_BRANCH_BIRTHS).permitted
