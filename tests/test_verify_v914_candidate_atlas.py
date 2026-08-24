import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/verify_v914_candidate_atlas.py"


def test_missing_prior_source_fails_with_actionable_message(tmp_path):
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--atlas", str(ROOT / "artifacts/v914_candidate_atlas"),
            "--historical-prior", str(tmp_path / "missing_historical.parquet"),
            "--cleavage-extension-prior", str(tmp_path / "missing_cleavage.parquet"),
        ],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "prior population source missing for historical_domain" in result.stderr
    assert "missing_historical.parquet" in result.stderr
    assert "--historical-prior" in result.stderr


def test_prior_paths_are_explicit_files_not_an_implicit_root_layout():
    source = SCRIPT.read_text()
    assert "prior_root" not in source
    assert "--historical-prior" in source and "--cleavage-extension-prior" in source
