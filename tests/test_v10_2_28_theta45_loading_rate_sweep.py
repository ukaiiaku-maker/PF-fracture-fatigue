from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from arrhenius_fracture.loading_rate_v10228 import (
    rate_tag,
    resolve_loading_rate,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_LAUNCHER = ROOT / "scripts" / "run_v10_2_28_paper_four_class_theta30_1000um.sh"
RATE_LAUNCHER = ROOT / "scripts" / "run_v10_2_28_paper_four_class_1000um_orientation_rate.sh"
SWEEP_LAUNCHER = ROOT / "scripts" / "run_v10_2_28_theta45_loading_rate_sweep.sh"
BUILDER_PATH = ROOT / "scripts" / "build_v10_2_28_rate_enabled_orientation_launcher.py"
SOURCE_SCHEDULER = ROOT / "scripts" / "run_v10_2_27_paper_four_class_30deg_long_rcurves.sh"
SOURCE_PLOTTER = ROOT / "scripts" / "plot_v10_2_27_paper_four_class_rcurves.py"


def _builder_module():
    spec = importlib.util.spec_from_file_location("rate_launcher_builder", BUILDER_PATH)
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


@pytest.mark.parametrize(
    ("factor", "expected_dt", "expected_rate", "expected_tag"),
    (
        (0.01, 840.0, 2.0e-7 / 840.0, "rate0p01x"),
        (1.0, 8.4, 2.0e-7 / 8.4, "rate1x"),
        (100.0, 0.084, 2.0e-7 / 0.084, "rate100x"),
    ),
)
def test_loading_rate_specification(factor, expected_dt, expected_rate, expected_tag):
    spec = resolve_loading_rate(factor)
    assert spec.loading_rate_factor == pytest.approx(factor)
    assert spec.nominal_dU_m == pytest.approx(2.0e-7)
    assert spec.base_dt_s == pytest.approx(8.4)
    assert spec.nominal_dt_s == pytest.approx(expected_dt)
    assert spec.nominal_opening_rate_m_per_s == pytest.approx(expected_rate)
    assert spec.rate_tag == expected_tag
    assert rate_tag(factor) == expected_tag


def test_loading_rate_rejects_nonpositive_values():
    for value in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            resolve_loading_rate(value)


def test_rate_and_sweep_launchers_have_valid_bash_syntax():
    for path in (RATE_LAUNCHER, SWEEP_LAUNCHER):
        completed = subprocess.run(
            ["bash", "-n", str(path)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr


def test_sweep_is_theta45_sequential_common_seed_design():
    source = SWEEP_LAUNCHER.read_text()
    required = (
        'RATE_FACTORS=${RATE_FACTORS:-"1 100 0.01"}',
        'THETA=${THETA:-45}',
        'MAX_JOBS=${MAX_JOBS:-1}',
        'if max_jobs != 1:',
        '"common_random_numbers_across_loading_rates": True',
        '"execution": "sequential_rates_and_one_case_at_a_time"',
        'bash scripts/run_v10_2_28_paper_four_class_1000um_orientation_rate.sh',
    )
    for token in required:
        assert token in source


def test_rate_builder_generates_valid_outer_and_case_scheduler(tmp_path: Path):
    builder = _builder_module()
    transformed = builder.transform(BASE_LAUNCHER.read_text())

    generated_outer = tmp_path / "rate_orientation.sh"
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
        "campaign_lock_v3",
        '"loading_rate_factor": float(os.environ["LOADING_RATE_FACTOR"])',
        '"nominal_dt_s": float(os.environ["DT_S"])',
        'LOADING_RATE_FACTOR="$LOADING_RATE_FACTOR"',
        'NOMINAL_OPENING_RATE_M_PER_S="$NOMINAL_OPENING_RATE_M_PER_S"',
    )
    for token in required_outer:
        assert token in transformed

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
        "v10.2.28_paper_four_class_orientation_loading_rate_campaign_v1",
        "v10.2.28_orientation_loading_rate_case_contract_v1",
        '"common_random_numbers_across_loading_rates": True',
        '"loading_rate_factor": float(os.environ["LOADING_RATE_FACTOR"])',
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


def test_rate_span_is_four_decades():
    slow = resolve_loading_rate(0.01).nominal_opening_rate_m_per_s
    fast = resolve_loading_rate(100.0).nominal_opening_rate_m_per_s
    assert math.log10(fast / slow) == pytest.approx(4.0)
