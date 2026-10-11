from arrhenius_fracture.coupled_void_execution_r3 import execute_coupled_void_r3


def test_clean_donor_executes_one_coupled_fracture_void_sequence(tmp_path):
    result = execute_coupled_void_r3(tmp_path / "coupled")
    assert result["passed"]
    assert result["qualification"]["one_global_event_scheduler"]
    assert not result["qualification"]["second_fracture_clock_in_void_adapter"]
    assert result["active_cavity"]["terminal_radius_m"] > 0.0
    assert result["active_cavity"]["event_count"] > 0
    assert result["common_state_sequence"]["no_void_fracture_intervals"] > 0
    assert result["common_state_sequence"]["void_enabled_fracture_intervals"] > 0
    assert result["fracture_adapter"]["actual_renewals"] > 0
    assert result["fracture_adapter"]["complete_state_parity"]
