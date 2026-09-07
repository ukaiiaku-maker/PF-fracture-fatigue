import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/A_native_analytical_overlay_all_1d_v1"


def table(name):
    with (OUT / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_manifest_is_prospective_and_parameter_free():
    data = json.loads((OUT / "analytical_input_manifest.json").read_text())
    assert data["created_before_loading_numerical_da_dN"] is True
    assert data["fitted_to_da_dN"] is False
    assert data["empirical_Paris_law_used"] is False
    assert data["predeclared_descriptive_error_bands_decade"] == [0.05, 0.1, 0.3]


def test_complete_inventory_and_qualification_counts():
    rows = table("physical_condition_inventory.csv")
    assert len(rows) == 146
    counts = {}
    for row in rows:
        counts[row["stationarity_classification"]] = counts.get(row["stationarity_classification"], 0) + 1
    assert counts == {
        "STEADY_STATE_QUALIFIED": 110,
        "NUMERICAL_VALIDATION_DUPLICATE": 27,
        "TRANSIENT_NOT_QUALIFIED": 9,
    }
    assert len({row["source_result_path"] for row in rows}) == len(rows)


def test_all_levels_and_source_hashes_are_present():
    rows = table("analytical_predictions.csv")
    for row in rows:
        assert row["source_hash"]
        assert row["A0_da_dN"]
        assert row["A1_da_dN"]
        assert row["A2_da_dN"]


def test_nonstationary_points_are_excluded_from_error_counts():
    summary = table("analytical_error_summary.csv")
    overall = [row for row in summary if row["grouping"] == "overall"]
    assert {row["model_level"] for row in overall} == {"A0", "A1", "A2"}
    assert {int(row["count"]) for row in overall} == {110}


def test_virtual_paths_and_figures_are_complete():
    assert len(table("CT_analytical_life_comparison.csv")) == 42
    figures = list((OUT / "figures").glob("*.png"))
    assert len(figures) == 9
    assert all(path.stat().st_size > 10_000 for path in figures)
