import importlib.util
import csv
from pathlib import Path

P = Path(__file__).parents[1] / "scripts/verify_v914_map_budget_prefix_identity.py"
S = importlib.util.spec_from_file_location("v", P)
M = importlib.util.module_from_spec(S)
S.loader.exec_module(M)


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_compare_prefix_identical_rows_pass():
    rows = [{"cumulative_cycles": "10.0", "K_MPa_sqrt_m": "20.0"}, {"cumulative_cycles": "20.0", "K_MPa_sqrt_m": "21.0"}]
    result = M.compare_prefix(rows, rows, ["cumulative_cycles", "K_MPa_sqrt_m"])
    assert result["identical"] is True
    assert result["compared_rows"] == 2


def test_compare_prefix_detects_first_divergence():
    a = [{"cumulative_cycles": "10.0"}, {"cumulative_cycles": "20.0"}]
    b = [{"cumulative_cycles": "10.0"}, {"cumulative_cycles": "999.0"}]
    result = M.compare_prefix(a, b, ["cumulative_cycles"])
    assert result["identical"] is False
    assert result["row_index"] == 1


def test_compare_prefix_only_checks_shared_length():
    # A larger-budget run having MORE rows than the smaller one is expected
    # and must not itself count as a divergence.
    a = [{"cumulative_cycles": "10.0"}]
    b = [{"cumulative_cycles": "10.0"}, {"cumulative_cycles": "20.0"}]
    result = M.compare_prefix(a, b, ["cumulative_cycles"])
    assert result["identical"] is True
    assert result["compared_rows"] == 1


def _write_common_fixture(dir_a, dir_b, *, events_a=None, events_b=None, extra_block_b=True):
    events = [{"cumulative_cycles": "10.0", "cycle_index": "10", "K_MPa_sqrt_m": "20.0", "threshold_action": "0.1", "physical_hazard_action": "0.1", "committed_advance_m": "1e-6", "cumulative_extension_m": "1e-6", "tip_radius_m": "1e-6", "mobile_total_m2": "1.0", "retained_total_m2": "0.0"}]
    blocks = [{"start_cycle": "0", "end_cycle": "10", "block_stride": "10", "numerical_pass": "True", "threshold_crossed": "True", "hazard_clock_action_before": "0.0", "hazard_threshold_action": "0.3"}]
    anchors = [{"cycle_index": "1", "hazard_action": "0.1", "shielding_MPa_sqrt_m": "0.0", "mobile_line_content": "1.0", "retained_line_content": "0.0", "returned_source_slip": "0.0", "physical_return_fraction": "0.0", "tip_radius_m": "1e-6", "cumulative_source_slip": "0.0", "raw_return_fraction": "0.0", "resolution": "exact_readiness_cycle", "hazard_clock_action": "0.1", "hazard_threshold_action": "0.3", "block_stride": "", "block_hazard_action": ""}]
    (dir_a / "run_contract.json").write_text(__import__("json").dumps({"candidate": "x", "physics_sha256": "abc", "repository_head": "old"}))
    (dir_b / "run_contract.json").write_text(__import__("json").dumps({"candidate": "x", "physics_sha256": "abc", "repository_head": "new"}))
    write_csv(dir_a / "events.csv", events_a if events_a is not None else events)
    write_csv(dir_b / "events.csv", events_b if events_b is not None else events + [{**events[0], "cumulative_cycles": "20.0"}])
    write_csv(dir_a / "block_history.csv", blocks)
    write_csv(dir_b / "block_history.csv", blocks + [blocks[0]] if extra_block_b else blocks)
    write_csv(dir_a / "anchor_history.csv", anchors)
    write_csv(dir_b / "anchor_history.csv", anchors + [anchors[0]])


def test_verify_pair_reports_identical_prefix(tmp_path):
    dir_a = tmp_path / "a"; dir_a.mkdir()
    dir_b = tmp_path / "b"; dir_b.mkdir()
    _write_common_fixture(dir_a, dir_b)
    result = M.verify_pair("4096", dir_a, "6144", dir_b)
    assert result["prefix_identical"] is True


def test_compare_run_contracts_ignores_repository_head():
    a = {"candidate": "x", "deltaK_MPa_sqrt_m": 16.2, "physics_sha256": "abc", "repository_head": "old"}
    b = {"candidate": "x", "deltaK_MPa_sqrt_m": 16.2, "physics_sha256": "abc", "repository_head": "new"}
    assert M.compare_run_contracts(a, b)["identical"] is True


def test_compare_run_contracts_flags_real_physics_divergence():
    a = {"candidate": "x", "physics_sha256": "abc", "repository_head": "old"}
    b = {"candidate": "x", "physics_sha256": "DIFFERENT", "repository_head": "new"}
    result = M.compare_run_contracts(a, b)
    assert result["identical"] is False
    assert result["field"] == "physics_sha256"


def test_verify_pair_includes_run_contract_and_anchor_checks(tmp_path):
    dir_a = tmp_path / "a"; dir_a.mkdir()
    dir_b = tmp_path / "b"; dir_b.mkdir()
    contract_a = {"candidate": "x", "physics_sha256": "abc", "repository_head": "old"}
    contract_b = {"candidate": "x", "physics_sha256": "abc", "repository_head": "new"}
    (dir_a / "run_contract.json").write_text(__import__("json").dumps(contract_a))
    (dir_b / "run_contract.json").write_text(__import__("json").dumps(contract_b))
    events = [{"cumulative_cycles": "10.0", "cycle_index": "10", "K_MPa_sqrt_m": "20.0", "threshold_action": "0.1", "physical_hazard_action": "0.1", "committed_advance_m": "1e-6", "cumulative_extension_m": "1e-6", "tip_radius_m": "1e-6", "mobile_total_m2": "1.0", "retained_total_m2": "0.0"}]
    blocks = [{"start_cycle": "0", "end_cycle": "10", "block_stride": "10", "numerical_pass": "True", "threshold_crossed": "True", "hazard_clock_action_before": "0.0", "hazard_threshold_action": "0.3"}]
    anchors = [{"cycle_index": "1", "hazard_action": "0.1", "shielding_MPa_sqrt_m": "0.0", "mobile_line_content": "1.0", "retained_line_content": "0.0", "returned_source_slip": "0.0", "physical_return_fraction": "0.0", "tip_radius_m": "1e-6", "cumulative_source_slip": "0.0", "raw_return_fraction": "0.0", "resolution": "exact_readiness_cycle", "hazard_clock_action": "0.1", "hazard_threshold_action": "0.3", "block_stride": "", "block_hazard_action": ""}]
    write_csv(dir_a / "events.csv", events)
    write_csv(dir_b / "events.csv", events)
    write_csv(dir_a / "block_history.csv", blocks)
    write_csv(dir_b / "block_history.csv", blocks)
    write_csv(dir_a / "anchor_history.csv", anchors)
    write_csv(dir_b / "anchor_history.csv", anchors)
    result = M.verify_pair("4096", dir_a, "6144", dir_b)
    assert result["prefix_identical"] is True
    assert result["run_contract"]["identical"] is True
    assert result["anchors"]["identical"] is True


def test_verify_pair_fails_on_anchor_divergence_even_if_events_match(tmp_path):
    dir_a = tmp_path / "a"; dir_a.mkdir()
    dir_b = tmp_path / "b"; dir_b.mkdir()
    (dir_a / "run_contract.json").write_text(__import__("json").dumps({"candidate": "x", "repository_head": "old"}))
    (dir_b / "run_contract.json").write_text(__import__("json").dumps({"candidate": "x", "repository_head": "new"}))
    events = [{"cumulative_cycles": "10.0", "cycle_index": "10", "K_MPa_sqrt_m": "20.0", "threshold_action": "0.1", "physical_hazard_action": "0.1", "committed_advance_m": "1e-6", "cumulative_extension_m": "1e-6", "tip_radius_m": "1e-6", "mobile_total_m2": "1.0", "retained_total_m2": "0.0"}]
    blocks = [{"start_cycle": "0", "end_cycle": "10", "block_stride": "10", "numerical_pass": "True", "threshold_crossed": "True", "hazard_clock_action_before": "0.0", "hazard_threshold_action": "0.3"}]
    anchor_a = [{"cycle_index": "1", "hazard_action": "0.1", "hazard_clock_action": "0.1", "hazard_threshold_action": "0.3", "shielding_MPa_sqrt_m": "0.0", "mobile_line_content": "1.0", "retained_line_content": "0.0", "returned_source_slip": "0.0", "physical_return_fraction": "0.0", "tip_radius_m": "1e-6", "cumulative_source_slip": "0.0", "raw_return_fraction": "0.0", "resolution": "exact_readiness_cycle", "block_stride": "", "block_hazard_action": ""}]
    anchor_b = [{**anchor_a[0], "hazard_clock_action": "999.0"}]
    write_csv(dir_a / "events.csv", events); write_csv(dir_b / "events.csv", events)
    write_csv(dir_a / "block_history.csv", blocks); write_csv(dir_b / "block_history.csv", blocks)
    write_csv(dir_a / "anchor_history.csv", anchor_a); write_csv(dir_b / "anchor_history.csv", anchor_b)
    result = M.verify_pair("4096", dir_a, "6144", dir_b)
    assert result["events"]["identical"] is True
    assert result["anchors"]["identical"] is False
    assert result["prefix_identical"] is False


def test_git_diff_is_empty_true_for_identical_ref(tmp_path):
    import subprocess
    repo = tmp_path / "repo"; repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "scripts").mkdir()
    (repo / "scripts" / "engine.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    result = M.git_diff_is_empty(repo, head, head, ["scripts/engine.py"])
    assert result["unchanged"] is True


def test_git_diff_is_empty_false_when_engine_file_changes(tmp_path):
    import subprocess
    repo = tmp_path / "repo"; repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "scripts").mkdir()
    (repo / "scripts" / "engine.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    old = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    (repo / "scripts" / "engine.py").write_text("x = 2\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "change engine"], cwd=repo, check=True)
    new = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    result = M.git_diff_is_empty(repo, old, new, ["scripts/engine.py"])
    assert result["unchanged"] is False


def test_external_engine_repo_fingerprint_reports_clean_and_head(tmp_path):
    import subprocess
    repo = tmp_path / "external"; repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "f.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    result = M.external_engine_repo_fingerprint(repo)
    assert result["clean"] is True
    assert len(result["head"]) == 40


def test_verify_pair_flags_diverging_prefix(tmp_path):
    dir_a = tmp_path / "a"; dir_a.mkdir()
    dir_b = tmp_path / "b"; dir_b.mkdir()
    events_a = [{"cumulative_cycles": "10.0", "cycle_index": "10", "K_MPa_sqrt_m": "20.0", "threshold_action": "0.1", "physical_hazard_action": "0.1", "committed_advance_m": "1e-6", "cumulative_extension_m": "1e-6", "tip_radius_m": "1e-6", "mobile_total_m2": "1.0", "retained_total_m2": "0.0"}]
    events_b = [{**events_a[0], "cumulative_cycles": "999.0"}]
    _write_common_fixture(dir_a, dir_b, events_a=events_a, events_b=events_b, extra_block_b=False)
    result = M.verify_pair("4096", dir_a, "6144", dir_b)
    assert result["prefix_identical"] is False
    assert result["events"]["identical"] is False
