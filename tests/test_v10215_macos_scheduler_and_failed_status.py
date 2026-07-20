from __future__ import annotations

import ast
from pathlib import Path


def test_main_scheduler_is_natively_macos_safe_and_clears_retry_markers():
    path = Path("scripts/run_v10_2_15_stage3_monotonic_temperature_sweep.sh")
    text = path.read_text()
    assert "if [[ ${#new_pids[@]} -gt 0 ]]" in text
    assert "PIDS=()" in text
    assert "LABELS=()" in text
    assert 'rm -f "$case_root/RUN_FAILED" "$case_root/exit_code.txt"' in text


def test_macos_wrapper_executes_native_safe_runner_without_runtime_rewrite():
    text = Path("scripts/run_v10_2_15_stage3_monotonic_temperature_sweep_macos.sh").read_text()
    assert 'exec bash "$ROOT/scripts/run_v10_2_15_stage3_monotonic_temperature_sweep.sh"' in text
    assert "text.replace" not in text


def test_overnight_launcher_uses_existing_2d_runner_directly():
    text = Path("scripts/run_v10_2_15_stage3_overnight.sh").read_text()
    assert "run_v10_2_15_stage3_monotonic_temperature_sweep.sh" in text
    assert "SIGNED_KERNEL_FAMILY_JSON" not in text
    assert "LOAD_INVARIANCE_ROOT" not in text


def test_status_reports_failed_case_log_tails():
    text = Path("scripts/status_v10_2_15_stage3.py").read_text()
    ast.parse(text)
    assert 'parser.add_argument("--failed-tail"' in text
    assert 'print("  failed cases:")' in text
    assert "failed_rows" in text
