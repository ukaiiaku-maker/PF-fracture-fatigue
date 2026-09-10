from arrhenius_fracture.r_tip_constitutive_handoff_v2 import audit


def test_existing_analytical_radius_law_is_distinguished_from_missing_child_handoff():
    result = audit()
    assert result["accepted_original_sharp_front_relation"]["exists"]
    assert result["R_TIP_OWNERSHIP"] == "PASS"
    assert result["R_TIP_DISTINCT_FROM_VOID_RADIUS"] == "PASS"
    assert result["R_TIP_CAUSAL_LAW"] == "NOT_DEFINED"
    assert not result["model_form_complete"]


def test_signed_prediction_and_reciprocal_control_are_frozen():
    result = audit()
    peers = result["signed_analytical_prediction_frozen_before_any_future_implementation"]
    assert [row["r_tip_over_r0"] for row in peers] == [0.75, 1.0, 1.25]
    assert peers[0]["uncapped_sigma_over_baseline_at_fixed_positive_K"] > 1.0
    assert peers[2]["uncapped_sigma_over_baseline_at_fixed_positive_K"] < 1.0
    assert result["consumer_graph"]["required_but_absent_edges"]


def test_consumer_graph_covers_both_original_hazards_and_event_selection():
    result = audit()
    nodes = {row["id"] for row in result["consumer_graph"]["nodes"]}
    assert "FrontEngine.lambda_cleave" in nodes
    assert "FrontEngine.lambda_emit" in nodes
    assert "_select_emitted_proposal" in nodes
    assert not result["child_continuation_path"]["emission_barrier_or_rate_consumed"]
