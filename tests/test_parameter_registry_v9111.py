from __future__ import annotations

import csv
import math

from arrhenius_fracture.material_manifest import MaterialManifest
from arrhenius_fracture.parameter_registry_v9111 import (
    CANONICAL_CANDIDATES,
    CANONICAL_STAGE3_OPTIONS,
    default_registry_path,
    select_option,
    stage3_case_rows,
    write_compatibility_manifest,
)


def test_canonical_stage3_fingerprints_and_grids():
    selected = {
        key: select_option(key, canonical_stage3_only=True)
        for key in CANONICAL_STAGE3_OPTIONS
    }
    assert {key: value.candidate_id for key, value in selected.items()} == CANONICAL_CANDIDATES
    assert selected["ceramic_primary"].mpz_length_um == 100.0
    assert selected["ceramic_primary"].mpz_n_bins == 200
    assert selected["weakT_primary"].mpz_length_um == 100.0
    assert selected["weakT_primary"].mpz_n_bins == 200
    assert selected["dbtt_primary"].mpz_length_um == 50.0
    assert selected["dbtt_primary"].mpz_n_bins == 80
    assert selected["peak_primary"].mpz_length_um == 50.0
    assert selected["peak_primary"].mpz_n_bins == 80


def test_exact_stage3_matrix_contains_40_unique_cases():
    rows = stage3_case_rows()
    assert len(rows) == 40
    keys = {(row["option_key"], row["temperature_K"]) for row in rows}
    assert len(keys) == 40
    assert {row["temperature_K"] for row in rows} == {
        float(value) for value in range(300, 1201, 100)
    }


def test_compatibility_manifest_preserves_selected_barriers(tmp_path):
    selected = select_option("peak_primary", canonical_stage3_only=True)
    path = write_compatibility_manifest(selected, tmp_path / "selected.csv")
    manifest = MaterialManifest.from_csv(path)
    assert manifest.name == "peak_primary"
    assert manifest.candidate_id == CANONICAL_CANDIDATES["peak_primary"]
    assert math.isclose(manifest.cleavage.G00_eV, float(selected.row["cleave_G00_eV"]))
    assert math.isclose(manifest.emission.G00_eV, float(selected.row["emit_G00_eV"]))
    assert math.isclose(manifest.peierls.H0_eV, float(selected.row["peierls_H0_eV"]))
    assert math.isclose(manifest.taylor.H0_eV, float(selected.row["taylor_H0_eV"]))
    assert manifest.max_K_shield_MPa_sqrt_m == 0.0
    with path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["option_key"] == "peak_primary"
    assert row["max_K_shield_MPa_sqrt_m"] == "0.0"


def test_packaged_registry_exists():
    assert default_registry_path().is_file()
