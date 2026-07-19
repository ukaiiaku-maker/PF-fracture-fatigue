from pathlib import Path
import ast
import importlib.util
import subprocess


def _load_wrapper():
    path = Path("scripts/run_v10_2_9_staged_parameterization.py")
    spec = importlib.util.spec_from_file_location("v1029_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_quality_diversity_python_wrapper_parses_and_imports():
    path = Path("scripts/run_v10_2_9_staged_parameterization.py")
    ast.parse(path.read_text())
    module = _load_wrapper()
    assert callable(module._quality_diverse_select)


def test_analytical_worker_flattens_temperature_trajectory(monkeypatch):
    module = _load_wrapper()

    def fake_worker(payload):
        return {
            "ok": True,
            "row": {
                "candidate_id": "C1",
                "target_class": "DBTT",
                "analytical_pass": True,
                "analytical_objective": 0.0,
            },
            "details": {
                "details": [
                    {
                        "temperature_K": 300.0,
                        "K_cleave_no_plastic_MPa_sqrt_m": 10.0,
                        "K_first_emission_MPa_sqrt_m": 11.0,
                        "emission_advantage_fraction": -0.1,
                        "linearized_source_bin_Kshield_MPa_sqrt_m": 0.2,
                        "mean_retained_fraction_indicator": 0.3,
                        "expected_source_activations": 4.0,
                        "expected_signed_line_content": -2.0,
                    }
                ]
            },
        }

    monkeypatch.setattr(module, "_ORIGINAL_ANALYTICAL_WORKER", fake_worker)
    result = module._flatten_analytical_worker(None)
    row = result["row"]
    assert row["analytical_K_cleave_300K"] == 10.0
    assert row["analytical_K_first_emission_300K"] == 11.0
    assert row["analytical_Kshield_300K"] == 0.2
    assert row["analytical_signed_line_300K"] == -2.0


def test_quality_diversity_shell_runner_has_valid_syntax():
    result = subprocess.run(
        ["bash", "-n", "scripts/run_v10_2_9_staged_parameterization.sh"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
