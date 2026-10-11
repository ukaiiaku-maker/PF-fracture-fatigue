import math

from scripts.consolidate_cavity_dwell import consolidate


def test_declared_dwell_is_an_exact_restart_chain_and_complete():
    result = consolidate()
    assert result["all_checkpoint_round_trips_exact"]
    assert result["all_suffix_handoffs_exact"]
    assert result["exposure_complete_within_binary_time_tolerance"]
    assert math.isclose(result["accepted_exposure_s"], 1.0e-5, abs_tol=2.0e-13)
    assert result["terminal_status"] == "requested_exposure_completed"
    assert result["terminal_unused_requested_duration_s"] == 0.0
    assert result["terminal_transport_rejection"] is None


def test_dwell_result_is_weak_evolution_not_interaction_completion():
    result = consolidate()
    disabled = result["observables"]["disabled"]
    enabled = result["observables"]["enabled"]
    assert disabled["graph_length_change_m"] == 0.0
    assert enabled["graph_length_change_m"] == 0.0
    assert disabled["void_event_count_end"] == disabled["void_event_count_start"]
    assert enabled["void_event_count_end"] == enabled["void_event_count_start"]
    assert enabled["cavity_radius_change_m"] > 0.0
    assert enabled["terminal_crack_void_connection"] is None
