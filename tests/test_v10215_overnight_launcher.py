from __future__ import annotations

from pathlib import Path
import subprocess


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
    assert "arrhenius_fracture.sharp_front_v10_1_7_5" in text
    assert "parameter_overlay_only" in text
    assert "signed_atlas_used" in text
    assert "SIGNED_KERNEL_FAMILY_JSON" not in text
    assert "LOAD_INVARIANCE_ROOT" not in text
    assert "ENGINE_CONFIG" not in text
    assert "build_v10_2_14" not in text
