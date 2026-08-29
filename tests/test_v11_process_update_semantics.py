import pytest

from arrhenius_fracture.process_update_semantics_v11 import classify_process_update


def test_disabled_legacy_hazard_checkpoint_sync_is_not_time_refinement():
    decision = classify_process_update(
        {
            "fired": True,
            "n_fire": 1,
            "stochastic_hazard_enabled": False,
            "physical_hazard_action_step": 0.0,
            "avalanche_checkpoint_synchronized": True,
            "checkpoint_committed_total_m": 20e-6,
        },
        directional_event_expected=False,
        permitted_physical_hazard_action=0.075,
    )
    assert decision.event_semantics == "process_checkpoint_synchronization"
    assert not decision.refinement_required
    assert decision.physical_hazard_action_step == 0.0


def test_genuine_legacy_physical_hazard_crossing_retains_refinement():
    decision = classify_process_update(
        {
            "fired": True,
            "n_fire": 1,
            "stochastic_hazard_enabled": True,
            "physical_hazard_action_step": 0.2,
            "avalanche_checkpoint_synchronized": True,
        },
        directional_event_expected=True,
        permitted_physical_hazard_action=0.075,
    )
    assert decision.event_semantics == "legacy_physical_hazard_crossing"
    assert decision.refinement_required
    assert decision.physical_hazard_action_step == pytest.approx(0.2)


def test_resolved_directional_clock_increment_is_ordinary_process_advance():
    decision = classify_process_update(
        {
            "fired": False,
            "n_fire": 0,
            "stochastic_hazard_enabled": False,
            "physical_hazard_action_step": 0.0,
            "avalanche_checkpoint_synchronized": True,
        },
        directional_event_expected=False,
        permitted_physical_hazard_action=0.15,
    )
    assert decision.event_semantics == "ordinary_process_state_advance"
    assert not decision.refinement_required


@pytest.mark.parametrize("rate", [1e-34, 1e-46, 1e-64])
def test_dormant_directional_rate_does_not_create_process_sentinel(rate):
    dH = rate * 8.4
    decision = classify_process_update(
        {
            "fired": False,
            "n_fire": 0,
            "stochastic_hazard_enabled": False,
            "physical_hazard_action_step": 0.0,
            "hazard_progress_rate": dH,
            "avalanche_checkpoint_synchronized": True,
        },
        directional_event_expected=False,
        permitted_physical_hazard_action=0.15,
    )
    assert not decision.refinement_required
    assert decision.physical_hazard_action_step == 0.0


def test_shared_owner_checkpoint_sync_preserves_membership_contract():
    member_tips = ("tip-a", "tip-b")
    decision = classify_process_update(
        {
            "fired": True,
            "n_fire": 1,
            "stochastic_hazard_enabled": False,
            "physical_hazard_action_step": 0.0,
            "avalanche_checkpoint_synchronized": True,
        },
        directional_event_expected=False,
        permitted_physical_hazard_action=0.075,
    )
    owner_by_tip = {tip: "shared-cluster" for tip in member_tips}
    assert not decision.refinement_required
    assert tuple(tip for tip, owner in owner_by_tip.items() if owner == "shared-cluster") == member_tips
