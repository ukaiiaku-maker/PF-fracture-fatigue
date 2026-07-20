from __future__ import annotations

from pathlib import Path
import subprocess


def test_stage3_runner_shell_syntax_and_matrix_defaults():
    root = Path(__file__).resolve().parents[1]
    runner = root / "scripts" / "run_v10_2_15_stage3_monotonic_temperature_sweep.sh"
    subprocess.run(["bash", "-n", str(runner)], check=True)
    text = runner.read_text()
    assert 'OPTIONS=${OPTIONS:-"ceramic_primary weakT_primary dbtt_primary peak_primary"}' in text
    assert 'TEMPS=${TEMPS:-"300 400 500 600 700 800 900 1000 1100 1200"}' in text
    assert 'TARGET_EXT_UM=${TARGET_EXT_UM:-500}' in text
    assert '--no-wake-shielding' in text
    assert '--mobile-shield-fraction 0' in text
    assert 'arrhenius_fracture.sharp_front_v10_2_15' in text
