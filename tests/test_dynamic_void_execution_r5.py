from arrhenius_fracture.dynamic_void_execution_r5 import execute_dynamic_void_r5


def test_recovered_dynamic_void_owner_uses_current_backend(tmp_path):
    result = execute_dynamic_void_r5(tmp_path / "dynamic")
    assert result["passed"]
    assert result["A_void_off"]["direct_backend_bitwise_parity"]
    assert result["B_native_incubation"]["vacancy_birth_clock_evaluated"]
    assert result["B_native_incubation"]["direct_birth_clock_evaluated"]
    assert result["C_active_dynamic"]["radius_changed"]
    assert result["C_active_dynamic"]["native_fracture_events"] >= 1
    assert result["transaction"]["rollback_exact"]
    assert not result["fixed_R4_observation"]["used_for_dynamic_run"]
