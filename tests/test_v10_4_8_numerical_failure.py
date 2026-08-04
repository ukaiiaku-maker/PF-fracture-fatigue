from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from arrhenius_fracture import (
    sharp_front_v10_4_8_numerical_failure_audited as v1048,
)


FIXED_POINT_MESSAGE = (
    "v10.4.3 mechanics/plasticity fixed point did not converge after adaptive "
    "timestep subdivision: T=300 K, step=135, dt_cur=8.4e-08 s, "
    "trial_fraction=1e-08, min_trial_fraction=1e-08, dt_retries=2, "
    "max_dt_retries=20, max_iterations=80, residual=8555.23, "
    "ep_residual=130.348, rho_residual=8555.23, relaxation=0.25"
)


def _load_script(name: str):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parser_accepts_only_the_production_fixed_point_failure():
    failure = v1048._parse_fixed_point_failure(RuntimeError(FIXED_POINT_MESSAGE))
    assert failure is not None
    assert failure.diagnostics["temperature_K"] == 300
    assert failure.diagnostics["step"] == 135
    assert failure.diagnostics["dt_cur_s"] == pytest.approx(8.4e-8)
    assert failure.diagnostics["trial_fraction"] == pytest.approx(1e-8)
    assert failure.diagnostics["min_trial_fraction"] == pytest.approx(1e-8)
    assert failure.diagnostics["residual"] == pytest.approx(8555.23)

    assert v1048._parse_fixed_point_failure(RuntimeError("unrelated")) is None


def test_writer_is_fail_closed(tmp_path):
    for name in (
        "COMPLETE",
        "PLASTIC_FLOW",
        "PLASTICITY_DOMINATED",
        "NUMERICAL_STAGNATION",
        "numerical_stagnation_audit.json",
        "stage3_case_status.json",
    ):
        (tmp_path / name).write_text("stale\n")

    failure = v1048._parse_fixed_point_failure(RuntimeError(FIXED_POINT_MESSAGE))
    assert failure is not None
    v1048._write_fixed_point_failure(tmp_path, failure)

    assert (tmp_path / "NUMERICAL_FIXED_POINT_FAILURE").is_file()
    audit_path = tmp_path / "numerical_fixed_point_failure_audit.json"
    assert audit_path.is_file()
    for name in (
        "COMPLETE",
        "PLASTIC_FLOW",
        "PLASTICITY_DOMINATED",
        "NUMERICAL_STAGNATION",
        "numerical_stagnation_audit.json",
        "stage3_case_status.json",
    ):
        assert not (tmp_path / name).exists()

    audit = json.loads(audit_path.read_text())
    assert audit["classification"] == "numerical_fixed_point_failure"
    assert audit["complete"] is False
    assert audit["plasticity_dominated"] is False
    assert audit["minimum_adaptive_trial_fraction_exhausted"] is True
    assert audit["exit_code"] == 5
    assert audit["step"] == 135
    assert audit["dt_cur_s"] == pytest.approx(8.4e-8)


def test_main_writes_fixed_point_audit_and_exits_five(monkeypatch, tmp_path):
    def fail(args):
        raise RuntimeError(FIXED_POINT_MESSAGE)

    monkeypatch.setattr(v1048._v1047, "main", fail)

    with pytest.raises(SystemExit) as caught:
        v1048.main(["--out", str(tmp_path)])

    assert caught.value.code == v1048.FIXED_POINT_FAILURE_EXIT_CODE
    assert (tmp_path / "NUMERICAL_FIXED_POINT_FAILURE").is_file()
    audit = json.loads(
        (tmp_path / "numerical_fixed_point_failure_audit.json").read_text()
    )
    assert audit["temperature_K"] == 300
    assert audit["step"] == 135


def test_main_does_not_relabel_unrelated_runtime_error(monkeypatch, tmp_path):
    def fail(args):
        raise RuntimeError("unrelated solver defect")

    monkeypatch.setattr(v1048._v1047, "main", fail)

    with pytest.raises(RuntimeError, match="unrelated solver defect"):
        v1048.main(["--out", str(tmp_path)])

    assert not (tmp_path / "NUMERICAL_FIXED_POINT_FAILURE").exists()


def test_scheduler_patcher_guards_nonzero_solver_exit(monkeypatch):
    patcher = _load_script("patch_v10_4_8_generated_scheduler.py")
    source = '''  echo "START: option=${option} T=${T}K seed=${case_seed} target=${TARGET_EXT_UM}um theta=${THETA}"
  env \\
    CLEAVAGE_HAZARD_SEED="$case_seed" \\
    PERSISTENT_SOURCE_MIN_WIDTH_UM="$PERSISTENT_SOURCE_MIN_WIDTH_UM" \\
    "${cmd[@]}" > "$log" 2>&1
  local rc=$?
  echo "$rc" > "$case_root/exit_code.txt"
  if [[ "$rc" -ne 0 ]]; then
    echo "simulation_exit_$rc" > "$case_root/RUN_FAILED"
  fi
'''
    monkeypatch.setattr(
        patcher,
        "_load_base_patcher",
        lambda: SimpleNamespace(transform=lambda text: text),
    )

    generated = patcher.transform(source)
    assert "local rc=0" in generated
    assert "if env \\" in generated
    assert '"${cmd[@]}" > "$log" 2>&1; then' in generated
    assert "else\n    rc=$?\n  fi" in generated
    assert 'echo "$rc" > "$case_root/exit_code.txt"' in generated
    assert 'echo "simulation_exit_$rc" > "$case_root/RUN_FAILED"' in generated
