import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
OUT = ROOT / "analysis_outputs/pf_current_source_general_multifront_v6_1"


def load(name):
    return json.loads((OUT / name).read_text())


def test_v6_source_patch_and_snapshots_are_commit_bound():
    provenance = load("pf_general_multifront_v6_1_source_provenance.json")
    assert provenance["v6_record_commit"] == "d993dd933f52281b89cf22e3aa9cf0db61d64487"
    assert provenance["v6_record_tree"] == "40aced628472af5803176fb0d8f87334d5fe63f5"
    assert provenance["v6_parent_commit"] == "ae9a06d8c42287428e917baef143b0c1142cefd8"
    assert provenance["v6_parent_tree"] == "2d0edbc7c7b51d81ab5678065ca68a1cef45d134"
    patch = ROOT / provenance["v6_binary_patch"]["path"]
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == provenance["v6_binary_patch"]["sha256"]
    assert not provenance["pre_v12_execution_file_changed_in_v6"]
    assert len(provenance["v6_source_snapshots"]) == 9
    for row in provenance["v6_source_snapshots"].values():
        path = ROOT / row["snapshot_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]


def test_call_graph_scheduler_handoff_and_multiowner_products_pass():
    call_graph = load("pf_general_multifront_production_call_graph_v6_1.json")
    assert call_graph["all_source_bindings_resolved"]
    assert call_graph["v12_source_physics_front_cap"] is None
    assert call_graph["v12_provider_limit_kind"] == "explicit_operational_resource_policy"
    scheduler = list(csv.DictReader(
        (OUT / "pf_general_multifront_scheduler_compatibility_v6_1.csv").open()
    ))
    assert len(scheduler) == 176
    assert any(row["step"] == "369" and row["action_type"] == "two_arm" for row in scheduler)
    assert {row["scheduler_policy"] for row in scheduler} == {
        "v11_correlated_proposal_compatibility", "v12_global_earliest_proposal",
    }
    handoff = list(csv.DictReader(
        (OUT / "pf_general_multifront_partial_handoff_v6_1.csv").open()
    ))
    assert len(handoff) == 6
    assert all(row["conserved"] == row["signed_conserved"] == "True" for row in handoff)
    interval = load("pf_general_multifront_multiowner_interval_v6_1.json")
    assert interval["qualification"] == "PASS"
    assert interval["active_front_count"] >= 4
    assert interval["unresolved_two_front_owner_count"] >= 1


def test_checkpoint_runtime_parity_and_no_solve_preflight_pass():
    checkpoint = load("pf_general_multifront_checkpoint_roundtrip_v6_1.json")
    assert checkpoint["qualification"] == "PASS"
    assert checkpoint["records"]["multiowner_n8"]["active_front_count"] == 8
    assert checkpoint["records"]["multiowner_n8"]["process_owner_count"] == 4
    parity = load("pf_general_multifront_v5_4_2_runtime_parity_v6_1.json")
    assert parity["qualification"] == "PASS"
    assert all(parity["gates"].values())
    assert parity["cases"]["theta40_corrected_enabled_max2_seed3621"][
        "step_369_renewal_distance_m"
    ] == 5.000000000000025e-06
    preflight = load("pf_general_multifront_production_preflight_v6_1.json")
    assert preflight["qualification"] == "PASS"
    for row in preflight["cases"].values():
        assert row["provider_lookup_count"] == row["mechanics_solve_count"] == 0
        assert not row["pf_worker_started"] and not row["fem_worker_started"]
        assert not row["production_output_root_created"]


def test_report_preserves_execution_boundary():
    report = (OUT / "PF_CURRENT_SOURCE_GENERAL_MULTIFRONT_V6_1.md").read_text()
    assert "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS" in report
    assert "production with more than two fronts remains not executed" in report
    assert "parent_process_zone_still_unresolved" in report
    assert "Predictive recursive-branching physics is not validated" in report
    validation = load("pf_general_multifront_validation_v6_1.json")
    assert validation["qualification"] == "PASS_RELATIVE_TO_ACCEPTED_SEVEN_LEGACY_FAILURE_BASELINE"
    assert validation["full_suite"]["new_failures"] == 0
    assert validation["full_suite"]["failed"] == 7
    assert validation["compileall"]["passed"] and validation["git_diff_check"]["passed"]
