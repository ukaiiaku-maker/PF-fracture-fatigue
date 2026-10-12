import pytest

from arrhenius_fracture.mechanical_correspondence_r5 import (
    execute_mechanical_correspondence_r5, signed_consumer_probe,
)


def test_actual_spatial_class_preserves_resolved_signs():
    result = signed_consumer_probe()
    assert result["mixed_sign_preserved"]
    assert result["negative_component_reaches_transport_consumer"]
    assert result["positive_reference_has_positive_minimum_velocity"]


def test_complete_mechanical_input_replay(tmp_path):
    result = execute_mechanical_correspondence_r5(
        "integration_evidence/r5_dynamic_void_state_correspondence/"
        "current_law_spatial_signed",
        tmp_path / "mechanical.json",
    )
    assert result["passed"]
    assert result["common_random_owner"]["forked_complete_initial_owner"]
    assert result["positive_loading_result_changed_after_sign_fix"]
    causal = result["causal_localization"]
    assert causal["first_divergent_accepted_update"] == "load_20um_geometry_commit"
    assert causal["common_event_owner"]["spatial_event_advance_m"] == pytest.approx(
        causal["common_event_owner"]["reduced_event_advance_m"]
    )
    assert causal["first_different_state_coordinate"]["spatial_path_angle_deg"] != 0.0
