from pathlib import Path

from scripts.build_dbtt_qualification_options import build, construct_rows


def test_dbtt_qualification_rows_preserve_distinct_lineages():
    rows = construct_rows()
    assert len(rows) == 2
    assert rows[0]["qualification_parent_option_key"].startswith(
        "v913_paper_dbtt01"
    )
    assert rows[1]["qualification_parent_option_key"] == "dbtt_primary"
    assert rows[0]["option_key"] != rows[1]["option_key"]
    assert all("taylor_0p10eV_qualification" in row["option_key"] for row in rows)


def test_dbtt_qualification_options_are_constructor_valid_and_unselected(tmp_path: Path):
    report = build(tmp_path / "options")
    assert report["status"] == "PASS"
    assert report["production_selection"] is None
    assert report["physical_calibration_claimed"] is False
    assert len(report["options"]) == 2
    assert all(row["domain"]["all_declared_points_admissible"] for row in report["options"])
    assert {row["source_lifecycle_provenance"] for row in report["options"]} == {
        "current_persistent_paper_lineage",
        "older_approved_finite_source_refresh_and_recovery_lineage",
    }
