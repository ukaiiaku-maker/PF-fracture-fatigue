import csv
import json
from pathlib import Path

from scripts import build_v10_2_30_refined_candidate_registry as builder


def test_candidate_registry_preserves_coordinates_and_fixes_common_physics(tmp_path):
    template = Path("arrhenius_fracture/data/materials/v10_2_27_paper_four_class_registry.csv")
    with template.open(newline="") as stream:
        template_row = next(csv.DictReader(stream))
    source_row = {"candidate_id": "candidate_A"}
    source_row.update({key: str(index + 0.25) for index, key in enumerate(builder.DIRECT_COORDINATES)})
    source_row["Tref_K"] = "300"
    for key in ("cleave_gT_eV_per_K", "emit_gT_eV_per_K", "cleave_sT_GPa_per_K", "emit_sT_GPa_per_K"):
        source_row[key] = "0"
    source = tmp_path / "source.csv"
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(source_row))
        writer.writeheader(); writer.writerow(source_row)
    registry, selection = tmp_path / "registry.csv", tmp_path / "selection.json"
    assert builder.main(["--source", str(source), "--candidate-id", "candidate_A",
                         "--template", str(template), "--out-registry", str(registry),
                         "--out-selection", str(selection)]) == 0
    with registry.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert all(row[key] == source_row[key] for key in builder.DIRECT_COORDINATES)
    assert row["source_sites_per_system"] == template_row["source_sites_per_system"]
    assert row["Tref_K"] == template_row["Tref_K"]
    assert row["n_bins_recommended"] == "80"
    payload = json.loads(selection.read_text())
    assert payload["common_physics_changed"] is False
    assert payload["legacy_m64_label_applicable"] is False
    assert payload["candidates"][0]["Tref_normalization_rate_invariant"] is True


def test_native_atlas_parquet_preserves_22d_vector_and_template_common_physics(tmp_path):
    import pandas as pd

    template = Path("arrhenius_fracture/data/materials/v10_2_27_paper_four_class_registry.csv")
    with template.open(newline="") as stream:
        template_row = next(csv.DictReader(stream))
    source_row = {"candidate_id": "V914_ATLAS_test"}
    source_row.update({key: index + 0.125 for index, key in enumerate(builder.ATLAS_COORDINATES)})
    source = tmp_path / "source.parquet"
    pd.DataFrame([source_row]).to_parquet(source, index=False)
    registry, selection = tmp_path / "registry.csv", tmp_path / "selection.json"
    assert builder.main(["--source", str(source), "--candidate-id", "V914_ATLAS_test",
                         "--template", str(template), "--out-registry", str(registry),
                         "--out-selection", str(selection)]) == 0
    with registry.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert all(float(row[key]) == source_row[key] for key in builder.ATLAS_COORDINATES)
    assert row["peierls_nu0_s"] == template_row["peierls_nu0_s"]
    assert row["taylor_nu0_s"] == template_row["taylor_nu0_s"]
    assert float(row["cleave_gT_eV_per_K"]) == 0.0
    payload = json.loads(selection.read_text())
    assert payload["candidates"][0]["implicit_temperature_neutral_coordinates"] is True


def test_candidate_registry_accepts_explicit_resolution(tmp_path):
    template = Path("arrhenius_fracture/data/materials/v10_2_27_paper_four_class_registry.csv")
    source_row = {"candidate_id": "candidate_A", "Tref_K": "300"}
    source_row.update({key: str(index + 0.25) for index, key in enumerate(builder.DIRECT_COORDINATES)})
    for key in ("cleave_gT_eV_per_K", "emit_gT_eV_per_K", "cleave_sT_GPa_per_K", "emit_sT_GPa_per_K"):
        source_row[key] = "0"
    source = tmp_path / "source.csv"
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(source_row))
        writer.writeheader(); writer.writerow(source_row)
    registry, selection = tmp_path / "registry.csv", tmp_path / "selection.json"
    assert builder.main(["--source", str(source), "--candidate-id", "candidate_A",
                         "--template", str(template), "--n-bins", "128",
                         "--out-registry", str(registry), "--out-selection", str(selection)]) == 0
    with registry.open(newline="") as stream:
        assert next(csv.DictReader(stream))["n_bins_recommended"] == "128"
    assert json.loads(selection.read_text())["numerical_bins"] == 128
