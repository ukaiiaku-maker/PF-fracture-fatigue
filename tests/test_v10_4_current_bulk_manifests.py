import csv
from pathlib import Path

from arrhenius_fracture.bulk_plasticity_manifest_v104 import BulkManifestParameters
from scripts.build_v10_4_current_bulk_manifests import SOURCE, build


ROOT = Path(__file__).resolve().parents[1]


def test_current_four_family_rows_project_exactly_into_native_bulk_manifests():
    report = build()
    assert report["status"] == "PASS"
    assert len(report["products"]) == 4
    with SOURCE.open(newline="") as stream:
        source = {row["option_key"]: row for row in csv.DictReader(stream)}
    for item in report["products"]:
        parsed = BulkManifestParameters.from_csv(ROOT / item["path"])
        assert parsed.option_key == item["option_key"]
        assert parsed.candidate_id == source[item["option_key"]]["candidate_id"]
        assert parsed.taylor_H0_eV == float(source[item["option_key"]]["taylor_H0_eV"])
        assert parsed.taylor_corr_rho_c_m2 == float(
            source[item["option_key"]]["taylor_corr_rho_c_m2"]
        )
