import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_packaged_campaign_runner_is_directly_invocable_from_repo_root():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "fracture_final_current/campaign/run_or_resume.py"),
            "--audit-only",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert '"passed": true' in result.stdout
