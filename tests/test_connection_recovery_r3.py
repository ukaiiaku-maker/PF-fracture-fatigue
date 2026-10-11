from arrhenius_fracture.connection_recovery_r3 import run_connection_recovery


def test_native_connection_veto_acceptance_and_restart_continuation(tmp_path):
    result = run_connection_recovery(tmp_path)
    assert result["passed"]
    assert not result["veto"]["trial"]["accepted"]
    assert result["veto"]["trial"]["energy_margin_J_per_m"] < 0.0
    assert result["veto"]["rollback_exact"]
    accepted = result["accepted"]
    assert accepted["trial"]["energy_margin_J_per_m"] > 0.0
    assert accepted["geometry_eligibility"]["remaining_ligament_m"] > 0.0
    assert accepted["connected"]["void_phase"] == "CONNECTED_VOID"
    assert accepted["checkpoint"]["restart_exact"]
    assert accepted["downstream_child"]["trial"]["accepted"]
    assert accepted["downstream_child"]["restart_exact"]
    assert accepted["downstream_continuation"]["trial"]["accepted"]
    assert accepted["downstream_continuation"]["restart_exact"]
    assert accepted["downstream_continuation"]["state"]["active_tip_ids"] == ["void-front-1"]
