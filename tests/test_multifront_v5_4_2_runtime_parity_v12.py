from pathlib import Path

from arrhenius_fracture.v5_4_2_runtime_parity_v12 import (
    replay_full_v5_4_2_terminal_runtime_parity,
    replay_v5_4_2_runtime_parity,
)


def test_authoritative_v5_4_2_transaction_and_runtime_parity(historical_product):
    result = replay_v5_4_2_runtime_parity(
        historical_product.root, source_root=historical_product.source_root,
    )
    assert result["qualification"] == "PASS"
    assert all(result["gates"].values())
    enabled = result["cases"]["theta40_corrected_enabled_max2_seed3621"]
    assert enabled["accepted_event_count"] == 84
    assert enabled["step_369_renewal_distance_m"] == 5.000000000000025e-06
    control = result["cases"]["theta40_corrected_control_max1_seed3621"]
    assert control["step_369_correlated_birth_exact"] is None
    assert control["step_369_correlated_birth_applicability"] == "not_applicable"
    assert control["step_369_branch_disabled_policy_behavior_exact"]


def test_complete_v5_4_2_terminal_transaction_and_runtime_parity():
    result = replay_full_v5_4_2_terminal_runtime_parity(Path(__file__).parents[1])
    assert result["qualification"] == "PASS"
    assert all(result["gates"].values())
    assert result["cases"]["control"]["accepted_event_count"] == 85
    assert result["cases"]["control"]["terminal_step"] == 662
    enabled = result["cases"]["enabled"]
    assert enabled["accepted_event_count"] == 86
    assert enabled["terminal_step"] == 813
    assert enabled["assessment_claimed_terminal_step"] == 807
    assert not enabled["assessment_terminal_step_matches_archive"]
    assert enabled["owner_transition_and_renewal_evidence"]["renewal_distance_m"] == (
        5.000000000000025e-06
    )
    assert enabled["owner_transition_and_renewal_evidence"]["daughter_front_ids"] == [
        "b0fa22bb892937f8", "b7d5efe0822562b9",
    ]


def test_control_step_369_uses_policy_field_not_correlated_birth_claim():
    result = replay_full_v5_4_2_terminal_runtime_parity(Path(__file__).parents[1])
    control = result["cases"]["control"]["branch_disabled_policy_evidence"]
    assert control["step_369_branch_disabled_policy_behavior_exact"]
    assert control["step_369_correlated_birth_exact"] is None
    assert control["step_369_correlated_birth_applicability"] == "not_applicable"
