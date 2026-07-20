from __future__ import annotations

import ast
from pathlib import Path


def test_macos_scheduler_patches_empty_array_reap_and_retry_markers():
    path = Path("scripts/run_v10_2_15_stage3_monotonic_temperature_sweep_macos.sh")
    text = path.read_text()
    assert "if [[ ${#new_pids[@]} -gt 0 ]]" in text
    assert "PIDS=()" in text
    assert "LABELS=()" in text
    assert 'rm -f "$case_root/RUN_FAILED" "$case_root/exit_code.txt"' in text


def test_overnight_launcher_uses_macos_safe_scheduler():
    text = Path("scripts/run_v10_2_15_stage3_overnight.sh").read_text()
    assert "run_v10_2_15_stage3_monotonic_temperature_sweep_macos.sh" in text


def test_status_reports_failed_case_log_tails():
    text = Path("scripts/status_v10_2_15_stage3.py").read_text()
    ast.parse(text)
    assert 'parser.add_argument("--failed-tail"' in text
    assert 'print("  failed cases:")' in text
    assert "failed_rows" in text
