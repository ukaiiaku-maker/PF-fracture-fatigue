from __future__ import annotations

from scripts.qualify_long_growth_analysis_contract import evidence_inventory, qualify


def test_long_growth_evidence_inventory_is_explicit():
    inventory = evidence_inventory()
    assert set(inventory["cases"]) == {"cold_300K", "hot_1200K"}
    for case in inventory["cases"].values():
        assert case["required_files"]


def test_retained_long_growth_contract_passes_when_evidence_is_available():
    inventory = evidence_inventory()
    if not inventory["available"]:
        # Portable clean-checkout behavior is fail-closed and deterministic;
        # full-release evidence availability is enforced by the command exit.
        report = qualify()
        assert report["decision"] == "INCOMPLETE_EXTERNAL_EVIDENCE_MISSING"
        assert report["retrieval_instruction"]
        return

    report = qualify()
    assert report["decision"] == "COMPLETE_RETAINED_LONG_GROWTH_ANALYSIS"
    assert report["contract_results"]["chronological_stage_history"] == "PASS"
    assert report["contract_results"]["output_policy_state_independence"] in {
        "PASS_NATIVE_BYTE_IDENTICAL_STATE", "PENDING_DETERMINISTIC_NATIVE_PAIR"
    }
    for case in report["cases"].values():
        assert case["accepted_event_pairs"] == 203
        assert case["terminal_projected_extension_m"] >= 1.0e-3
        assert case["terminal_path_arclength_m"] >= case["terminal_projected_extension_m"]
        assert case["repeated_stage_coordinate_count"] > 0
        assert all(not query["interpolation_performed"] for query in case["length_queries"])
