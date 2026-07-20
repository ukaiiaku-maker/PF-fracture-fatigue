from __future__ import annotations

import ast
from pathlib import Path
import subprocess


def test_campaign_atlas_builder_parses():
    path = Path("scripts/build_v10_2_14_campaign_ready_active_only_atlas.py")
    ast.parse(path.read_text())
    text = path.read_text()
    assert "mechanics_rerun_performed\": False" in text
    assert "material_parameter_refit_performed\": False" in text
    assert "campaign_parameterization_allowed\": True" in text


def test_overnight_launcher_has_valid_shell_syntax_and_full_mode():
    path = Path("scripts/run_v10_2_15_stage3_overnight.sh")
    result = subprocess.run(
        ["bash", "-n", str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    text = path.read_text()
    assert "MODE=full" in text
    assert "MAX_JOBS=${MAX_JOBS:-2}" in text
    assert "TARGET_EXT_UM=${TARGET_EXT_UM:-500}" in text
