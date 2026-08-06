import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from arrhenius_fracture.crack_network_v11 import CrackNetworkState
from arrhenius_fracture.directional_competition_v11 import (
    DirectionalCompetitionState,
    accept_reservation,
    commit_directional_interval,
    competition_state_from_dict,
    competition_state_from_json,
    competition_state_to_dict,
    competition_state_to_json,
    construct_action_proposals,
    preview_directional_interval,
    release_reservation,
    reserve_action,
    tungsten_cleavage_candidates,
)


ROOT = Path(__file__).parents[1]


def evolved_state():
    candidates = tungsten_cleavage_candidates(theta_deg=30.0)
    state = DirectionalCompetitionState.initialize(candidates, global_hazard_seed=3621)
    hazards = []
    for index, hazard in enumerate(reversed(state.hazard_states)):
        preview = preview_directional_interval(
            hazard, lambda_per_s=2.0 + index, start_time_s=5.0, duration_s=1.2
        )
        hazards.append(commit_directional_interval(hazard, preview))
    return replace(state, hazard_states=tuple(reversed(hazards)))


def test_round_trip_and_deterministic_bytes_for_multiple_pending_events():
    state = evolved_state()
    encoded = competition_state_to_json(state)
    restored = competition_state_from_json(encoded)
    assert restored == state
    assert competition_state_to_json(restored) == encoded


def test_active_released_and_accepted_reservations_round_trip():
    state = evolved_state()
    proposal = construct_action_proposals(state.hazard_states, correlation_interval_s=1.0)[0]
    active = reserve_action(
        state, proposal, event_rewards_m=(2e-6,) * len(proposal.member_event_ids)
    )
    released = release_reservation(active, proposal.action_id)
    assert competition_state_from_json(competition_state_to_json(active)) == active
    assert competition_state_from_json(competition_state_to_json(released)) == released

    fresh = reserve_action(
        state,
        next(item for item in construct_action_proposals(
            state.hazard_states, correlation_interval_s=1.0
        ) if item.action_id != proposal.action_id),
        event_rewards_m=(1e-6,),
    )
    accepted = accept_reservation(fresh, fresh.reservations[0].action_id)
    assert competition_state_from_json(competition_state_to_json(accepted)) == accepted


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda data: data.update(schema="unknown"), "unsupported"),
        (lambda data: data["hazard_states"][0].update(candidate_id="unknown"), "candidate|belongs"),
        (
            lambda data: data["hazard_states"][0]["pending_events"].append(
                copy.deepcopy(data["hazard_states"][0]["pending_events"][0])
            ),
            "duplicate",
        ),
    ],
)
def test_malformed_serialization_fails_closed(mutation, message):
    payload = competition_state_to_dict(evolved_state())
    mutation(payload)
    with pytest.raises(ValueError, match=message):
        competition_state_from_dict(payload)


def test_legacy_one_tip_state_bytes_and_reference_hash_are_unchanged():
    one_tip = CrackNetworkState.one_tip([(0.0, 0.0), (5.0e-6, 0.0)])
    assert hashlib.sha256(one_tip.to_json().encode()).hexdigest() == (
        "1c5025e451d4c667ac51288e45d00c9e27b42411c35874eab3ff9b13585bed5a"
    )
    reference = ROOT / "v11_branching_disabled_baseline.json"
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == (
        "fbd35339b09b685e7a524447b4e6414b1b3364c3cb7f7c012b478365f02af191"
    )
    payload = json.loads(reference.read_text())
    for relative, expected in payload["output_sha256"].items():
        runtime = ROOT / payload["output_directory"] / relative
        if runtime.is_file():
            assert hashlib.sha256(runtime.read_bytes()).hexdigest() == expected
