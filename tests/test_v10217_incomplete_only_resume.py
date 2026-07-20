from pathlib import Path
import subprocess


def test_incomplete_only_resume_shell_parses_and_uses_result_classification():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "resume_v10_2_17_incomplete_only.sh"
    completed = subprocess.run(
        ["bash", "-n", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    text = path.read_text()
    assert "classify_v10_2_15_stage3_case.py" in text
    assert "stage3_case_status.json" in text
    assert 'payload.get("complete") is True' in text
    assert "SKIP verified complete" in text
    assert "resume_incomplete_plan.tsv" in text
    assert "incomplete_backup_" in text
    assert "ALLOW_PARTIAL=1" in text
    assert "SKIP_FINISHED=0" in text
    assert "NO_PLOTS=\"$NO_PLOTS\"" in text


def test_classifier_reads_solver_plain_header_crack_path():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts" / "classify_v10_2_15_stage3_case.py").read_text()
    assert "np.genfromtxt" in text
    assert "names=True" in text
    assert "crack_path_file" in text
    assert 'case_root / f"crack_path_{int(round(temperature_K))}K.csv"' in text
