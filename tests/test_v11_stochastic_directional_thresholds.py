import math
import random

import numpy as np
import pytest

from arrhenius_fracture.directional_competition_v11 import (
    DirectionalCompetitionState,
    commit_directional_interval,
    competition_state_from_json,
    competition_state_to_json,
    preview_directional_interval,
    tungsten_cleavage_candidates,
)


def first_threshold(seed, candidates=None):
    inventory = candidates or tungsten_cleavage_candidates(theta_deg=30.0)
    state = DirectionalCompetitionState.initialize(inventory, global_hazard_seed=seed)
    return {item.candidate_id: item.current_threshold_action for item in state.hazard_states}


def test_unit_exponential_threshold_statistics_and_survival():
    candidate = tungsten_cleavage_candidates(theta_deg=30.0)[0]
    samples = np.array([
        first_threshold(seed, (candidate,))[candidate.candidate_id]
        for seed in range(10000)
    ])
    assert samples.mean() == pytest.approx(1.0, abs=0.025)
    for time in (0.25, 0.5, 1.0, 2.0):
        assert np.mean(samples > time) == pytest.approx(math.exp(-time), abs=0.015)


def test_thresholds_are_candidate_order_invariant_seed_reproducible_and_restart_exact():
    candidates = list(tungsten_cleavage_candidates(theta_deg=30.0, include_110=True))
    expected = first_threshold(3621, candidates)
    random.Random(91).shuffle(candidates)
    assert first_threshold(3621, candidates) == expected
    assert first_threshold(3621, candidates) != first_threshold(3622, candidates)
    state = DirectionalCompetitionState.initialize(candidates, global_hazard_seed=3621)
    restored = competition_state_from_json(competition_state_to_json(state))
    assert restored == state


def test_stochastic_threshold_crossing_preserves_integrated_hazard_overshoot():
    candidate = tungsten_cleavage_candidates(theta_deg=30.0)[0]
    state = DirectionalCompetitionState.initialize((candidate,), global_hazard_seed=17)
    hazard = state.hazard_states[0]
    end_action = hazard.current_threshold_action + 0.4
    preview = preview_directional_interval(
        hazard, lambda_per_s=1.0, start_time_s=3.0, duration_s=end_action,
    )
    committed = commit_directional_interval(hazard, preview)
    assert preview.completed_events
    assert committed.action == pytest.approx(end_action)
    assert committed.action > preview.completed_events[0].action_after
    assert committed.current_threshold_action > committed.action
