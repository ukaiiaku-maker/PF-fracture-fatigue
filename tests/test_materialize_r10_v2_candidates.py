import csv
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_exact_reconstruction_matches_promoted_07172(tmp_path):
    out = tmp_path / "registry.csv"
    subprocess.run([
        sys.executable, str(ROOT / "scripts/materialize_r10_v2_candidates.py"),
        "--original-registry", str(ROOT / "integration_evidence/r10_joint_search_continuation/inputs/original_four_registry.csv"),
        "--parent-registry", str(ROOT / "integration_evidence/r10_joint_search_continuation/inputs/fatigue_parent_registry.csv"),
        "--anchor-registry", str(ROOT / "integration_evidence/r10_opening_alternatives/state_informed_48_registry.csv"),
        "--candidate-ids", "R10F_V2_07172", "--out", str(out),
    ], check=True, capture_output=True, text=True)
    rebuilt = {row["candidate_id"]: row for row in csv.DictReader(out.open())}
    current_path = ROOT / "runs/r10_12h_autonomous/acquisition_round_0_1024/native_information_registry.csv"
    current = {row["candidate_id"]: row for row in csv.DictReader(current_path.open())}
    for field in ("complete_row_sha256", "cohesive_parameter_sha256",
                  "plasticity_physics_sha256", "cohesive_parameter_vector",
                  "parent_complete_parameter_vector"):
        assert rebuilt["R10F_V2_07172__MS"][field] == current["R10F_V2_07172__MS"][field]
