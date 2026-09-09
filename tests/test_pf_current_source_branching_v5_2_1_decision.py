import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.publish_pf_current_source_branching_v5_2_1_decision import (
    strict_below_target_events,
)


ROOT = Path(__file__).resolve().parents[1]
V5 = ROOT / "analysis_outputs/pf_current_source_branching_corrected_v5_2_execution_20260831T190053Z"
OUT = ROOT / "analysis_outputs/pf_current_source_branching_corrected_v5_2_1_decision"
OLD_COVERAGE = ROOT / "analysis_outputs/pf_current_source_branching_capability/theta40_signed_kernel_cache/adb7754436a66542a38c17d671bc62639939d85075168a5db721b93b791e87d0/portable_assembly/coverage_audit.json"
FILES = (
    "PF_CURRENT_SOURCE_BRANCHING_CORRECTED_V5_2_1_DECISION.md",
    "pf_branching_corrected_v5_2_1_decision.json",
    "pf_branching_signed_kernel_coverage_failure_audit.json",
    "pf_branching_terminal_topology_coverage_bound.json",
)


def test_strict_below_target_event_semantics():
    assert strict_below_target_events(5.4241391411108, 3.8302222155948944) == 1
    assert strict_below_target_events(204.46317713942403, 3.2139380484327082) == 63
    assert strict_below_target_events(300.0, 3.2139380484327082) == 93
    assert strict_below_target_events(0.0, 3.0) == 0


def test_superseding_decision_separates_terminal_and_morphology_axes():
    decision = json.loads((OUT / "pf_branching_corrected_v5_2_1_decision.json").read_text())
    assert decision["pair_terminal_result"] == "CORRECTED_THETA40_REPLAY_STOPPED_FAIL_CLOSED_SIGNED_KERNEL_ENVELOPE"
    assert decision["corrected_morphology_capability"] == "CORRECTED_CURRENT_SOURCE_BRANCHING_MORPHOLOGY_CAPABILITY_DEMONSTRATED_BEFORE_ENVELOPE_STOP"
    assert decision["bounded_300um_pair_completion"] == "NOT_COMPLETED_SIGNED_KERNEL_ENVELOPE"
    assert decision["predictive_branching_physics_validated"] is False
    assert decision["all_eleven_morphology_success_criteria_pass"] is True
    assert len(decision["morphology_success_criteria"]) == 11
    assert all(item["pass"] for item in decision["morphology_success_criteria"])
    assert all("300" not in item["name"] for item in decision["morphology_success_criteria"])


def test_coverage_failure_and_terminal_topology_bounds():
    coverage = json.loads((OUT / "pf_branching_signed_kernel_coverage_failure_audit.json").read_text())
    assert coverage["old_planner_reproduction"]["required_um"] == pytest.approx(410.0009075646826)
    assert coverage["old_planner_reproduction"]["measured_endpoint_um"] == pytest.approx(415.0)
    assert coverage["observed_control"]["accepted_event_count"] == 84
    assert coverage["observed_control"]["accepted_events_by_angle_deg"] == {"-50": 40, "40": 44}
    assert coverage["observed_control"]["either_permitted_next_event_crosses_target"] is True
    assert coverage["corrected_single_front_semantics"]["maximum_pre_event_shared_extension_query_um"] == 465.0
    assert coverage["corrected_single_front_semantics"]["one_topology_quantum_guard_endpoint_um"] == 470.0

    bound = json.loads((OUT / "pf_branching_terminal_topology_coverage_bound.json").read_text())
    assert bound["one_below_target_long_arm_event_remains_possible"] is True
    assert bound["short_arm_below_target_events_remaining"] == 63
    assert bound["maximum_pre_event_shared_extension_query_um"] == pytest.approx(740.0)
    assert bound["measured_endpoint_with_one_quantum_guard_um"] == pytest.approx(745.0)
    assert bound["remaining_branch_capacity"] == 0
    assert "not a universal" in bound["scope"]
    assert bound["shared_process_extension_um"] != bound["total_network_new_geometry_um"]
    assert bound["shared_process_extension_um"] != bound["maximum_forward_reach_um"]


def test_v5_2_predecessor_is_immutable_and_publisher_is_deterministic(tmp_path):
    old_decision = V5 / "final_two_axis_decision.json"
    assert hashlib.sha256(old_decision.read_bytes()).hexdigest() == "ef517a314f393dab7079d55d20d8949819bfc1e47a02b4dd1213648f05f433f8"
    raw = json.loads((V5 / "raw_tree_freeze_manifest.json").read_text())
    assert raw["tree_sha256"] == "cec0e2523bd16ce18b541c1eb7cdf65ee26ba553b5ecfb657da617ed2321565e"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/publish_pf_current_source_branching_v5_2_1_decision.py"),
            "--v5-result",
            str(V5),
            "--old-coverage-audit",
            str(OLD_COVERAGE),
            "--out",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    for name in FILES:
        assert (tmp_path / name).read_bytes() == (OUT / name).read_bytes()
