import csv
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
OUT = ROOT / "analysis_outputs/pf_current_source_general_multifront_v6_2"


def load(name):
    return json.loads((OUT / name).read_text())


def test_required_v6_2_products_exist_and_preserve_boundary():
    names = {
        "PF_CURRENT_SOURCE_GENERAL_MULTIFRONT_V6_2.md",
        "pf_general_multifront_v6_2_source_provenance.json",
        "pf_general_multifront_current_source_hook_map_v6_2.json",
        "pf_general_multifront_engine_restore_v6_2.json",
        "pf_general_multifront_event_consumption_v6_2.csv",
        "pf_general_multifront_exact_trial_commit_v6_2.json",
        "pf_general_multifront_accepted_state_rollover_v6_2.json",
        "pf_general_multifront_daughter_competition_v6_2.json",
        "pf_general_multifront_checkpoint_roundtrip_v6_2.json",
        "pf_general_multifront_production_dry_run_v6_2.json",
    }
    assert {path.name for path in OUT.iterdir()} == names
    report = (OUT / "PF_CURRENT_SOURCE_GENERAL_MULTIFRONT_V6_2.md").read_text()
    assert "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS" in report
    assert "concrete_current_source_production_adapter`: `FAIL_CLOSED" in report
    assert "generic_N1_N2_PF_reproduction`: `NOT_EXECUTED" in report


def test_source_and_hook_records_are_fail_closed_and_exactly_bound():
    source = load("pf_general_multifront_v6_2_source_provenance.json")
    assert source["v6_1_1_record_commit"] == "cfc107ac88b7e8415b8fe6ae951bb9129c5bdef5"
    assert source["v6_1_1_record_tree"] == "46bb3c192915bdb08ba6c93adf0c5b75cdd7a369"
    assert source["v6_1_1_archive_sha256"] == "7291ef4522d173f039ada04f977ac4b0f65389337104a90962b8dac06c2bf973"
    hooks = load("pf_general_multifront_current_source_hook_map_v6_2.json")
    assert hooks["leaf_binding_qualification"] == "PASS"
    assert hooks["composition_qualification"] == "FAIL_CLOSED"
    assert all(row["status"] == "QUALIFIED" for row in hooks["hooks"].values())


def test_engine_restore_and_dry_run_are_zero_time_zero_rng_zero_solve():
    engine = load("pf_general_multifront_engine_restore_v6_2.json")
    assert engine["qualification"] == "PASS"
    assert all(engine["gates"].values())
    assert set(engine["cases"]) == {"N1_step1", "N2_postbirth", "N2_terminal"}
    dry = load("pf_general_multifront_production_dry_run_v6_2.json")
    assert dry["qualification"] == "FAIL_CLOSED"
    assert dry["provider_lookup_count"] == dry["mechanics_solve_count"] == 0
    assert not dry["pf_worker_started"] and not dry["fem_worker_started"]


def test_event_and_transaction_records_do_not_claim_physical_execution():
    with (OUT / "pf_general_multifront_event_consumption_v6_2.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert {row["sentinel"] for row in rows} >= {
        "no_event_interval", "accepted_one_arm", "step369_atomic_two_arm_birth",
        "rejected_trial", "configured_resource_stop", "clipped_or_coalesced_trial",
    }
    assert all(row["provider_solve_count"] == "0" for row in rows)
    exact = load("pf_general_multifront_exact_trial_commit_v6_2.json")
    assert not exact["independent_proposal_endpoint_reconstruction"]
    assert exact["production_hook_composition"] == "FAIL_CLOSED_NOT_EXECUTED"
