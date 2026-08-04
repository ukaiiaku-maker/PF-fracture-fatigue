from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "runs/v10_2_30_four_class_qualification_7a5133f_20260804"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_extension_matrix_and_real_source_checkpoints_are_fail_closed():
    module = load("extension", ROOT / "scripts/v10230_developed_extension_supervisor.py")
    rows = module.matrix()
    assert len(rows) == 4
    assert {row["fraction"] for row in rows} == {0.95}
    inspected = [module.inspect_source(CAMPAIGN, row) for row in rows]
    assert all(row["starting_extension_um"] < 100.0 for row in inspected)
    assert all(row["rng_state_present"] for row in inspected)
    assert all(row["physical_hazard_action"] == 0.0 for row in inspected)
    assert [row["starting_event_count"] for row in inspected] == [6, 4, 5, 6]


def test_developed_analyzer_retains_complete_interval_physics(tmp_path):
    module = load("analysis", ROOT / "scripts/analyze_v10_2_30_developed_extension.py")
    assert module.main([str(CAMPAIGN), "--out", str(tmp_path)]) == 0
    payload = json.loads((tmp_path / "production_summary.json").read_text())
    assert payload["case_count"] == 4
    assert payload["event_interval_count"] == 21
    assert all(case["full_trajectory_da_dN_m_per_cycle"] > 0 for case in payload["cases"])
    assert (tmp_path / "complete_event_intervals.csv").is_file()
    assert (tmp_path / "restart_checkpoint_provenance.json").is_file()
