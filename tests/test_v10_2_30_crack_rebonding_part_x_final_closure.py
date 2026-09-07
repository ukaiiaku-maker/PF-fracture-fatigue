"""Final closure H: focused tests for the final-closure builders and the
comprehensive terminal verifier. All of these are pure reductions of
already-tracked artifacts -- fast, no physical simulation, no runs/
dependency.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"


def _run(script_name: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script_name)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


def test_admissibility_map_disposition_counts():
    proc = _run("build_part_x_admissibility_map.py")
    assert proc.returncode == 0, proc.stderr
    d = json.loads((ARTIFACTS_DIR / "part_x_admissibility_map.json").read_text())
    disp = d["disposition_counts"]
    assert disp["ADMITTED"] == 126
    assert disp["INTERRUPTED_NOT_SCIENCE"] == 3
    assert disp["INVALIDATED_DWELL_DURATION_WEIGHTING_BUG"] == 2
    assert disp["SUPERSEDED_WALL_BUDGET_TOO_SMALL"] == 2
    assert d["n_total_rows"] == 165


def test_producer_launch_provenance_closure_zero_diverged():
    proc = _run("build_part_x_producer_launch_provenance_closure.py")
    assert proc.returncode == 0, proc.stderr
    d = json.loads((ARTIFACTS_DIR / "part_x_producer_launch_provenance_closure.json").read_text())
    assert d["n_diverged"] == 0
    assert d["overall_decision"] == "PHYSICS_SOURCE_IDENTICAL_ALLOW_HEAD_DRIFT_CLOSED_FOR_ALL_PAIRS"
    assert d["n_producer_launch_pairs"] == 5
    for pair in d["pairs"]:
        assert pair["admissibility_decision"] == "PHYSICS_SOURCE_IDENTICAL_ALLOW_HEAD_DRIFT_CLOSED"


def test_local_slope_curvature_table_covers_5_protocols():
    proc = _run("build_part_x_local_slope_curvature_table.py")
    assert proc.returncode == 0, proc.stderr
    rows = list(csv.DictReader(open(ARTIFACTS_DIR / "px4_local_slope_curvature_table.csv")))
    protocols = {r["protocol"] for r in rows}
    assert protocols == {"D1", "D2", "D3", "D5", "D6_conditional_persistent"}
    assert len(rows) == 20  # 4 intervals x 5 protocols


def test_artifact_manifest_no_vague_missing_status():
    proc = _run("build_part_x_artifact_manifest.py")
    assert proc.returncode == 0, proc.stderr
    d = json.loads((ARTIFACTS_DIR / "part_x_artifact_manifest.json").read_text())
    valid_statuses = {
        "PRESENT_AND_VERIFIED", "GENERATED_IN_FINAL_CLOSURE",
        "NOT_APPLICABLE_WITH_REASON", "NOT_ARCHIVED_WITH_EXPLICIT_LIMITATION",
    }
    for r in d["requirements"]:
        assert r["status"] in valid_statuses, f"{r['id']} has invalid status {r['status']}"
        if r["status"] in ("NOT_APPLICABLE_WITH_REASON", "NOT_ARCHIVED_WITH_EXPLICIT_LIMITATION"):
            assert r["reason"], f"{r['id']} is {r['status']} but has no reason"


def test_figure_manifest_all_19_resolved():
    proc = _run("build_part_x_figure_manifest.py")
    assert proc.returncode == 0, proc.stderr
    d = json.loads((ARTIFACTS_DIR / "part_x_figure_manifest.json").read_text())
    assert d["all_19_resolved"] is True
    assert d["n_requirements"] == 19
    canonical_files = [r["canonical_file"] for r in d["requirements"]]
    assert len(canonical_files) == len(set(canonical_files))
    for r in d["requirements"]:
        full_path = REPO_ROOT / r["canonical_file"]
        assert full_path.is_file() and full_path.stat().st_size > 0


def test_final_decision_status_and_classification():
    d = json.loads((ARTIFACTS_DIR / "part_x_final_decision.json").read_text())
    assert d["status"] == "FINAL"
    assert d["primary_classification"] == "SIGNED_K_CRACK_REBONDING_PART_X_COMPLETE"
    assert d["subordinate_verification_classification"] == "PART_X_PX0_THROUGH_PX6_COMPLETE"
    # px4/px5 component decisions must be preserved unchanged (still provisional).
    px4d = json.loads((ARTIFACTS_DIR / "px4_scientific_decision.json").read_text())
    px5d = json.loads((ARTIFACTS_DIR / "px5_scientific_decision.json").read_text())
    assert px4d["status"] == d["supersedes"]["px4_scientific_decision.json"]["original_status_preserved_unchanged"]
    assert px5d["status"] == d["supersedes"]["px5_scientific_decision.json"]["original_status_preserved_unchanged"]


def test_D3_vs_D6_wording_is_not_order_of_magnitude():
    d = json.loads((ARTIFACTS_DIR / "px4_scientific_decision.json").read_text())
    text = d["interpretation"]["D3_vs_D6_persistence"]
    assert "6.18-fold" in text
    assert "NOT a full order of magnitude" in text
    # the corrected text should not assert "an order-of-magnitude difference" as fact
    assert "an order-of-magnitude difference driven" not in text


def test_terminal_verifier_passes():
    proc = _run("verify_part_x_terminal.py")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    d = json.loads((ARTIFACTS_DIR / "part_x_terminal_verification.json").read_text())
    assert d["overall_pass"] is True
    assert d["classification"] == "SIGNED_K_CRACK_REBONDING_PART_X_COMPLETE"
    failed = [k for k, v in d["checks"].items() if not v]
    assert not failed, f"failed checks: {failed}"


def test_terminal_verifier_portable_with_runs_hidden():
    import shutil

    run_root = REPO_ROOT / "runs" / "crack_rebonding_part_x_v1"
    if not run_root.is_dir():
        return  # already hidden/absent in this environment -- nothing to test
    parent = run_root.parent
    hidden = parent / "_test_hidden_runs_crack_rebonding_part_x_v1"
    shutil.move(str(run_root), str(hidden))
    try:
        proc = _run("verify_part_x_terminal.py")
        assert proc.returncode == 0, proc.stdout + proc.stderr
    finally:
        shutil.move(str(hidden), str(run_root))
    assert run_root.is_dir()
