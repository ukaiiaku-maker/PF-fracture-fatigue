import ast
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def test_high_cycle_runner_shell_syntax_and_contract():
    runner = ROOT / "scripts" / "run_v10_2_29_high_cycle_fatigue.sh"
    subprocess.run(["bash", "-n", str(runner)], check=True)
    text = runner.read_text()
    assert "sharp_front_v10_2_29_fixed_deltaK" in text
    assert 'HORIZONS=${HORIZONS:-"1e6 1e9 1e12"}' in text
    assert "DEVELOPED_START_UM" in text
    assert "extract_v10_2_29_developed_fatigue_growth.py" in text


def test_high_cycle_analysis_scripts_parse():
    for name in (
        "extract_v10_2_29_developed_fatigue_growth.py",
        "analyze_v10_2_29_developed_fatigue_campaign.py",
        "analyze_v10_2_29_horizon_scaling.py",
    ):
        path = ROOT / "scripts" / name
        ast.parse(path.read_text())
