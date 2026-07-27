from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BASE_LAUNCHER = (
    ROOT / "scripts" / "run_v10_2_28_paper_four_class_theta30_1000um.sh"
)
BUILDER = ROOT / "scripts" / "build_v10_2_30_rate_enabled_orientation_launcher.py"
RATE_LAUNCHER = ROOT / "scripts" / "run_v10_2_30_paper_four_class_orientation_rate.sh"
SWEEP = ROOT / "scripts" / "run_v10_2_30_theta45_loading_rate_sweep.sh"
SCREEN = ROOT / "scripts" / "run_v10_2_30_theta45_four_class_gate_screen.sh"
SOURCE_SCHEDULER = (
    ROOT / "scripts" / "run_v10_2_27_paper_four_class_30deg_long_rcurves.sh"
)
SOURCE_PLOTTER = ROOT / "scripts" / "plot_v10_2_27_paper_four_class_rcurves.py"
MODEL_ENTRY = (
    "arrhenius_fracture."
    "sharp_front_v10_2_30_hazard_energy_gated_audited"
)


def _builder_module():
    spec = importlib.util.spec_from_file_location("v10230_rate_builder", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scheduler_adapter_source(launcher: str) -> str:
    marker = 'OUTROOT="$OUTROOT" "$PYTHON_BIN" - <<\'PY\'\n'
    start = launcher.index(marker, launcher.index('SOURCE_SCHEDULER="$source_scheduler"'))
    start += len(marker)
    end = launcher.index('\nPY\n\nchmod +x "$generated_scheduler"', start)
    return launcher[start:end]


def test_v10230_shell_launchers_have_valid_syntax():
    for path in (RATE_LAUNCHER, SWEEP, SCREEN):
        completed = subprocess.run(
            ["bash", "-n", str(path)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, f"{path}: {completed.stderr}"


def test_v10230_builder_generates_outer_and_case_scheduler(tmp_path: Path):
    transformed = _builder_module().transform(BASE_LAUNCHER.read_text())

    generated_outer = tmp_path / "v10230_orientation.sh"
    generated_outer.write_text(transformed)
    syntax = subprocess.run(
        ["bash", "-n", str(generated_outer)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr

    required_outer = (
        "v10.2.30_hazard_energy_gated_orientation_rate_lock_v1",
        MODEL_ENTRY,
        '"hazard_energy_gate": True',
        '"absolute_athermal_Gc": False',
        '"gate_resolution": "every_internal_Strang_microstep"',
        '"fixed_DeltaK_energy_scaling": "(K_event/K_probe)^2"',
        "TARGET_EXT_UM must be finite and positive",
        "v10_2_30_hazard_energy_gate_campaign_lock.json",
    )
    for token in required_outer:
        assert token in transformed
    assert "fixed to 1000 um crack extension" not in transformed

    generated_scheduler = tmp_path / "generated_scheduler.sh"
    generated_plotter = tmp_path / "generated_plotter.py"
    outroot = tmp_path / "out"
    outroot.mkdir()
    environment = dict(os.environ)
    environment.update(
        {
            "SOURCE_SCHEDULER": str(SOURCE_SCHEDULER),
            "SOURCE_PLOTTER": str(SOURCE_PLOTTER),
            "GENERATED_SCHEDULER": str(generated_scheduler),
            "GENERATED_PLOTTER": str(generated_plotter),
            "OUTROOT": str(outroot),
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", _scheduler_adapter_source(transformed)],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout

    scheduler = generated_scheduler.read_text()
    required_scheduler = (
        "set -euo pipefail",
        MODEL_ENTRY,
        "v10.2.30_hazard_energy_gated_orientation_rate_campaign_v1",
        "v10.2.30_hazard_energy_gated_orientation_rate_case_contract_v1",
        '"hazard_energy_gate": True',
        '"absolute_athermal_Gc": False',
        '"gate_resolution": "every_internal_Strang_microstep"',
        "v10_2_30_hazard_energy_gate_audit.json",
        "stochastic_avalanche_geometry_events.json",
        "energy_available_integrated_J_per_m",
        "energy_dissipated_integrated_J_per_m",
        "dissipated > available + tolerance",
        'f"--dU {os.environ[\'DU_M\']}"',
        'f"--dt {os.environ[\'DT_S\']}"',
        '--dU "$DU_M" --dt "$DT_S" --n-stagger 2',
    )
    for token in required_scheduler:
        assert token in scheduler

    scheduler_syntax = subprocess.run(
        ["bash", "-n", str(generated_scheduler)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert scheduler_syntax.returncode == 0, scheduler_syntax.stderr


def test_four_class_screen_is_narrow_and_uses_audited_options():
    source = SCREEN.read_text()
    required = (
        "TARGET_EXT_UM=20",
        "TEMPS=900",
        "THETA=45",
        "LOADING_RATE_FACTOR=1",
        "MAX_JOBS=1",
        "v913_paper_peak01_0242980_persistent_sites",
        "v913_paper_dbtt01_0202500_persistent_sites",
        "v913_paper_weakT01_0129902_persistent_sites",
        "v913_paper_ceramic01_0077080_persistent_sites",
        "v10_2_30_hazard_energy_gate_audit.json",
        "stochastic_avalanche_geometry_events.json",
        "event_energy_balance_pass",
    )
    for token in required:
        assert token in source


def test_production_sweep_preserves_three_rates_common_seeds_and_four_options():
    source = SWEEP.read_text()
    required = (
        'RATE_FACTORS=${RATE_FACTORS:-"1 100 0.01"}',
        'THETA=${THETA:-45}',
        'TARGET_EXT_UM=${TARGET_EXT_UM:-1000}',
        'MAX_JOBS=${MAX_JOBS:-1}',
        '"common_random_numbers_across_loading_rates": True',
        '"execution": "sequential_rates_and_one_case_at_a_time"',
        '"model_entry": (',
        '"arrhenius_fracture."',
        '"sharp_front_v10_2_30_hazard_energy_gated_audited"',
        '"hazard_energy_gate": True',
        '"absolute_athermal_Gc": False',
        "run_v10_2_30_paper_four_class_orientation_rate.sh",
    )
    for token in required:
        assert token in source
