#!/usr/bin/env python3
"""Read-only checks of all saved physical parent inputs and companion outputs."""
import argparse
import json
from pathlib import Path
import pickle
import subprocess
import sys

from scripts.run_pf_current_source_multifront_field_atlas_v12 import CASES, atomic_json, sha256
from scripts.v13_value_fingerprint import physical_state_fingerprint as fp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.root / "companions/summary.json").read_text())
    assert len(result["cases"]) == len(CASES) == 8
    checks = []
    for case in CASES:
        parent = args.root / "physical_parents" / case / "parent"
        record = json.loads((parent / "parent_record.json").read_text())
        assert record["fresh_initialization"] and not record["historical_process_state_used"]
        assert sha256(parent / "event_context.pkl") == record["event_context_sha256"]
        payload = pickle.loads((parent / "event_context.pkl").read_bytes())
        for name, field in (("pre_cleavage", "pre_cleavage_checkpoint_sha256"),
                            ("accepted_single", "accepted_single_checkpoint_sha256")):
            manifest = json.loads((parent / f"{name}.json").read_text())
            assert sha256(parent / manifest["state_file"]) == manifest["state_sha256"] == record[field]
        accepted = payload["accepted_single_checkpoint"]
        assert fp(accepted.shared_process_state) == record["accepted_process_sha256"]
        assert fp(accepted.state.competition) == record["cleavage_rng_and_clocks_sha256"]
        assert fp(accepted.shared_process_state["engine_fields"]["_hazard_rng"]) == record["process_rng_sha256"]
        assert accepted.state.competition.consumed_event_ids == (record["event_id"],)
        assert record["parent_call_count_in_accepted_event_interval"] == 1
        companion_path = args.root / "companions" / case / "companion_qualification.json"
        companion = json.loads(companion_path.read_text())
        assert companion["clean_parent_record"] == record
        for observation in companion["companions"]:
            assert observation["single_fallback_sha256"] == fp(accepted.state)
            assert observation["post_primary_process_sha256"] == record["accepted_process_sha256"]
            assert observation["fixed_parent_endpoint_s"] == accepted.physical_time_s
            assert observation["fixed_opening_m"] == accepted.accepted_load
            assert observation["baseline_invariants_exact"]
            assert observation["global_time_increment_s"] == observation["new_baseline_rng_draws"] == 0
        for row in companion["sensitivity_surface"]:
            assert 0 <= row["P_embryo"] <= 1
            assert row["P_committed"] == (row["P_embryo"] if row["exact_pair_admissible"] else 0)
        checks.append({"case": case, "status": "PASS", "parent_record_sha256": sha256(parent / "parent_record.json"),
            "companion_record_sha256": sha256(companion_path), "analytic_grid_rows": len(companion["sensitivity_surface"])})
    modules = ["test_v13_clean_parent_capture", "test_v13_physical_companion_contract",
        "test_topology_transaction_v11", "test_directional_competition_transactions_v11",
        "test_multifront_checkpoint_output_v12"]
    pytest = subprocess.run([sys.executable, "-m", "pytest", "-q",
        *(f"tests/{name}.py" for name in modules)], text=True, capture_output=True, check=True)
    assert "35 passed" in pytest.stdout
    compile_result = subprocess.run([sys.executable, "-m", "compileall", "-q", "arrhenius_fracture",
        "scripts/run_v13_clean_parents.py", "scripts/qualify_v13_physical_companions.py",
        "scripts/report_v13_physical_qualification.py", "scripts/package_v13_physical_review.py",
        "scripts/verify_v13_physical_records.py"], capture_output=True, text=True, check=True)
    diff = subprocess.run(["git", "diff", "--check"], capture_output=True, text=True, check=True)
    atomic_json(args.root / "verification.json", {
        "producer_code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "record_integrity": checks,
        "focused_tests": {"passed": 35, "failed": 0,
            "modules": modules, "stdout": pytest.stdout, "stderr": pytest.stderr, "exit_code": pytest.returncode},
        "compileall": "PASS", "git_diff_check": "PASS",
        "compileall_exit_code": compile_result.returncode, "diff_check_exit_code": diff.returncode,
        "full_suite_repeated": False, "sampler_only_Monte_Carlo_repeated": False})
    print("Eight immutable clean-parent/companion record checks PASS")


if __name__ == "__main__":
    main()
