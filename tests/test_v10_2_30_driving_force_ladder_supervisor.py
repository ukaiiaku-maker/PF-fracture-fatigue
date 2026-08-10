from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import runpy


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "v10230_driving_force_ladder_supervisor.py"


def load_module():
    spec = importlib.util.spec_from_file_location("driving_force_ladder", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matrix_has_exact_peak_dbtt_ladder_and_dimensional_contract():
    module = load_module()
    rows = module.matrix()
    assert len(rows) == 6
    assert [row["fraction"] for row in rows] == [.95, .95, .975, .975, 1.0, 1.0]
    assert {row["label"] for row in rows} == {"peak", "dbtt"}
    assert {row["seed"] for row in rows if row["label"] == "peak"} == {1720}
    assert {row["seed"] for row in rows if row["label"] == "dbtt"} == {1001723}
    for row in rows:
        assert row["deltaK_MPa_sqrt_m"] == row["reference_deltaK_MPa_sqrt_m"] * row["fraction"]
        assert row["Kmax_MPa_sqrt_m"] == row["deltaK_MPa_sqrt_m"] / .9
        assert row["Kmin_MPa_sqrt_m"] == .1 * row["Kmax_MPa_sqrt_m"]
        assert row["frequency_Hz"] == 1000.0
        assert row["temperature_K"] == 300.0


def test_prepare_is_clean_atomic_and_non_overwriting(tmp_path, monkeypatch):
    module = load_module()
    monkeypatch.setattr(module.q, "free_gib", lambda _path: 20.0)
    monkeypatch.setattr(module.subprocess, "check_output", lambda command, **_kwargs:
                        "" if command[1:3] == ["status", "--porcelain"] else
                        ("test-head\n" if command[1:3] == ["rev-parse", "HEAD"] else "test-branch\n"))
    root = tmp_path / "ladder"
    payload = module.prepare(root)
    assert payload["launch_git_head"] == "test-head"
    assert module.validate_staged(root) == payload
    for row in module.matrix():
        status = json.loads((root / row["case"] / "qualification_status.json").read_text())
        assert status["status"] == "pending"
    import pytest
    with pytest.raises(RuntimeError, match="already exists"):
        module.prepare(root)


def test_run_concurrency_is_explicitly_bounded_to_one_or_two():
    module = load_module()
    assert module.parser().parse_args(["run", "/tmp/x"]).max_jobs == 2
    assert module.parser().parse_args(["run", "/tmp/x", "--max-jobs", "1"]).max_jobs == 1


def test_above_one_specialization_changes_only_fraction_matrix_and_manifest():
    namespace = runpy.run_path(str(
        ROOT / "scripts" / "v10230_above_one_driving_force_ladder_supervisor.py"
    ), run_name="above_one_ladder_test")
    module = namespace["ladder"]
    rows = module.matrix()
    assert [row["fraction"] for row in rows] == [1.025, 1.025, 1.05, 1.05]
    assert {row["label"] for row in rows} == {"peak", "dbtt"}
    assert module.MANIFEST_NAME == "above_one_driving_force_ladder_matrix.json"


def test_f1p100_transition_specialization_is_single_deliberate_level():
    namespace = runpy.run_path(str(
        ROOT / "scripts" / "v10230_f1p100_driving_force_ladder_supervisor.py"
    ), run_name="f1p100_ladder_test")
    module = namespace["ladder"]
    rows = module.matrix()
    assert [row["fraction"] for row in rows] == [1.1, 1.1]
    assert {row["label"] for row in rows} == {"peak", "dbtt"}
    assert module.MANIFEST_NAME == "f1p100_driving_force_ladder_matrix.json"
