from arrhenius_fracture.matched_execution_r6 import (
    accepted_increment_dataflow, execute_matched_r6,
)


def test_dataflow_keeps_bulk_and_tip_state_distinct():
    rows = accepted_increment_dataflow()
    bulk, tip = rows[:2]
    assert bulk["consumed_once"] and bulk["distinct_from_tip_mpz"]
    assert tip["consumed_once"] and tip["vacancy_bulk_strain_alias_forbidden"]


def test_matched_current_arms_use_same_schedule_and_restart(tmp_path):
    result = execute_matched_r6(tmp_path / "matched.json")
    assert result["passed"]
    assert result["qualification"]["passive_matches_no_void"]
    assert result["qualification"]["event_and_later_interval_executed"]
    assert result["qualification"]["complete_restart_exact"]
    assert result["qualification"]["moving_cavity_response_qualified"] is False
