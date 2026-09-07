from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from unittest.mock import patch


def _module():
    path = Path(__file__).parents[1] / "scripts" / "run_v10_2_30_A_native_plus_8PT_developed_panel.py"
    spec = spec_from_file_location("a8pt_panel", path)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_one_does_not_precreate_fresh_output_directory(tmp_path):
    mod = _module()
    root = tmp_path / "study"
    root.mkdir()
    (root / "A_native_plus_8PT_registry.csv").write_text("option_key\nA_NATIVE\n")
    (root / "A_native_plus_8PT_selection.json").write_text("{}\n")
    out = root / "developed" / "n80" / "A_NATIVE" / "DK_10.8"
    row = {
        "composite_id": "test", "parameter_option": "A_NATIVE",
        "deltaK_MPa_sqrt_m": 10.8, "result_path": str(out),
    }

    def fake_run(*args, **kwargs):
        assert out.parent.is_dir()
        assert not out.exists(), "qualified launcher requires a virgin output path"
        class Result: returncode = 2
        return Result()

    with patch.object(mod.subprocess, "run", side_effect=fake_run):
        result = mod.run_one(row, root, root / "jobs.csv")
    assert result["status"] == "LAUNCH_PREFLIGHT_FAILURE"

