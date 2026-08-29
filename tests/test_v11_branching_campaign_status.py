import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("status_v11", ROOT / "scripts/status_v11_branching_campaign.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_status_requires_valid_completion_marker_and_reports_branching(tmp_path):
    (tmp_path / "launcher.json").write_text(json.dumps({"git_head": "abc", "planned_cases": ["a", "b"]}))
    for name in ("a", "b"): (tmp_path / name).mkdir()
    (tmp_path / "a" / "run_complete.json").write_text(json.dumps({"schema": "v11.branching-run-complete/1", "status": "target_reached"}))
    (tmp_path / "a" / "branch_events.csv").write_text("header\nrow\n")
    (tmp_path / "b" / "run_complete.json").write_text("{}")
    status = module.campaign_status(tmp_path)
    assert status["completed_cases"] == 1; assert status["branching_cases"] == 1; assert status["planned_cases"] == 2
