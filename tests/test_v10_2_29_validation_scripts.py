from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_v10_2_29_300K_validation.sh"
GATE = ROOT / "scripts" / "run_v10_2_29_300K_growth_gate.sh"


def test_validation_shell_scripts_have_valid_syntax():
    for path in (RUNNER, GATE):
        completed = subprocess.run(
            ["bash", "-n", str(path)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr


def test_validation_scope_is_300K_and_uses_correct_rows():
    source = RUNNER.read_text()
    assert "--temperatures 300" in source
    assert "v913_paper_dbtt01_0202500_persistent_sites" in source
    assert "v913_paper_weakT01_0129902_persistent_sites" in source
    assert "sharp_front_v10_2_28_audited" in source
    assert "sharp_front_v10_2_29_fatigue_audited" in source
    assert "--fatigue-cycles" in source
    assert "--fatigue-hold-load" in source


def test_growth_gate_requires_a_committed_event():
    source = GATE.read_text()
    assert "extract_v10_2_29_fatigue_growth.py" in source
    assert "--require-event" in source
    assert "fatigue_event_growth_0300K.csv" in source
