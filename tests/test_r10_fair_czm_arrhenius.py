import json
from pathlib import Path


def test_original_mechanics_contract_is_fail_closed(tmp_path):
    from scripts.execute_r10_fair_czm_arrhenius import main
    assert main(["--out",str(tmp_path)]) == 0
    contract=json.loads((tmp_path/"COMMON_MODEL_CONTRACT.json").read_text())
    assert contract["mechanical_coordinates"]["separation_coordinate"] is None
    assert contract["full_czm_execution_status"] == "INTERFACE_GAP_UNRESOLVED"
    assert "k_env" in contract["excluded_environment"]["formula"]
    decision=json.loads((tmp_path/"CAPABILITY_DECISION.json").read_text())
    assert decision["capability_answer"] == "UNRESOLVED_FOR_FULL_CZM_IN_ORIGINAL_MECHANICS"
    assert decision["universal_czm_negative"] is False


def test_nonthermal_fixed_threshold_does_not_redraw(tmp_path):
    from scripts.execute_r10_fair_czm_arrhenius import main
    main(["--out",str(tmp_path)])
    text=(tmp_path/"NONTHERMAL_STATISTICAL_CLOSURE.csv").read_text()
    assert "NO_AUTONOMOUS_WAITING" in text
    assert "LOAD_HISTORY_EVENT_NOT_CYCLE_LIFETIME" in text
