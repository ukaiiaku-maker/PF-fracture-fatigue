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
