import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_release_builder_is_declarative_and_writes_all_products(tmp_path):
    out = tmp_path / "release"
    subprocess.run([sys.executable, "scripts/build_common_framework_release.py", "--out", str(out)], cwd=ROOT, check=True)
    required = {
        "FRAMEWORK_STATUS.json", "PROSPECTIVE_CAMPAIGN.jsonl", "PROSPECTIVE_CAMPAIGN.csv",
        "CAMPAIGN_SUMMARY.json", "MATERIAL_DOMAIN_AUDIT.csv", "COHESIVE_BENCHMARK.csv",
        "SOURCE_IDENTITIES.json", "COMMON_FRAMEWORK_SCIENTIFIC_SPECIFICATION.md", "COMMANDS.md",
    }
    assert required <= {path.name for path in out.iterdir()}
    summary = json.loads((out / "CAMPAIGN_SUMMARY.json").read_text())
    assert summary["case_count"] == 644
    assert summary["production_execution_authorized"] is False


def test_campaign_runner_refuses_execution():
    text = (ROOT / "scripts/run_prospective_common_campaign.py").read_text()
    assert "production execution is disabled" in text
    assert "subprocess" not in text


def test_no_stateful_kitagawa_in_new_runtime_products():
    paths = [
        ROOT / "arrhenius_fracture/prospective_campaign.py",
        ROOT / "arrhenius_fracture/cohesive_landscape.py",
        ROOT / "arrhenius_fracture/candidate_drive_contract.py",
        ROOT / "scripts/run_prospective_common_campaign.py",
    ]
    assert all("kitagawa" not in path.read_text().lower() for path in paths)
