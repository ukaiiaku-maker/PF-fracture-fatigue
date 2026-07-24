from __future__ import annotations

import csv
import json
from pathlib import Path

from arrhenius_fracture.sharp_front_v10_2_25 import (
    DEFAULT_REGISTRY,
    SELECTION_RECORD,
    VALID_OPTIONS,
)


ACTIVE_FIELDS = (
    "Tref_K",
    "cleave_G00_eV",
    "cleave_gT_eV_per_K",
    "cleave_sigc0_GPa",
    "cleave_sT_GPa_per_K",
    "cleave_exp_a",
    "cleave_exp_n",
    "cleave_floor_frac",
    "emit_G00_eV",
    "emit_gT_eV_per_K",
    "emit_sigc0_GPa",
    "emit_sT_GPa_per_K",
    "emit_exp_a",
    "emit_exp_n",
    "emit_floor_frac",
    "peierls_H0_eV",
    "peierls_activation_entropy_kB",
    "peierls_exp_a",
    "peierls_exp_n",
    "peierls_nu0_s",
    "taylor_H0_eV",
    "taylor_activation_entropy_kB",
    "taylor_exp_a",
    "taylor_exp_n",
    "taylor_nu0_s",
    "rho_source0_m2",
    "taylor_corr_rho_c_m2",
    "taylor_corr_scale",
    "c_blunt",
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_paper_campaign_selection_has_five_primary_and_one_control() -> None:
    payload = json.loads(SELECTION_RECORD.read_text())
    primary = payload["primary_candidates"]
    secondary = payload["secondary_candidates"]
    assert len(primary) == 5
    assert len(secondary) == 1
    assert payload["counts"] == {
        "primary_peak_like": 3,
        "primary_classic_dbtt_upper_shelf": 2,
        "secondary_rehardening_control": 1,
        "total": 6,
    }
    assert [row["candidate_id"] for row in primary] == [
        "v913_zeroD_sobol_0242980",
        "v913_zeroD_sobol_0127508",
        "v913_zeroD_sobol_0115460",
        "v913_zeroD_sobol_0202500",
        "v913_zeroD_sobol_0088403",
    ]
    assert [row["candidate_id"] for row in secondary] == [
        "v913_zeroD_sobol_0086420"
    ]
    assert secondary[0]["K50_2d_rehardening"] is True
    assert all(row["intended_topology_retained"] is True for row in primary)


def test_combined_registry_matches_selection_and_source_active_parameters() -> None:
    rows = _read(DEFAULT_REGISTRY)
    assert len(rows) == 6
    assert {row["option_key"]: row["candidate_id"] for row in rows} == VALID_OPTIONS

    material_dir = DEFAULT_REGISTRY.parent
    source_rows = _read(
        material_dir / "v10_2_23_v913_top10_persistent_site_registry.csv"
    ) + _read(
        material_dir / "v10_2_24_v913_top10_upper_shelf_registry.csv"
    )
    source_by_candidate = {row["candidate_id"]: row for row in source_rows}

    for row in rows:
        source = source_by_candidate[row["candidate_id"]]
        for field in ACTIVE_FIELDS:
            assert float(row[field]) == float(source[field]), (
                row["candidate_id"],
                field,
                row[field],
                source[field],
            )
        for field in (
            "source_recovery_rate_s",
            "retained_recovery_rate_s",
            "source_refresh_length_um",
            "recovery_nu0_s",
            "legacy_source_sites_active",
            "legacy_source_refresh_active",
            "explicit_recovery_active",
        ):
            assert float(row[field]) == 0.0


def test_paper_campaign_options_are_unique_and_stable() -> None:
    assert VALID_OPTIONS == {
        "v913_paper_peak01_0242980_persistent_sites": "v913_zeroD_sobol_0242980",
        "v913_paper_peak02_0127508_persistent_sites": "v913_zeroD_sobol_0127508",
        "v913_paper_peak03_0115460_persistent_sites": "v913_zeroD_sobol_0115460",
        "v913_paper_dbtt01_0202500_persistent_sites": "v913_zeroD_sobol_0202500",
        "v913_paper_dbtt02_0088403_persistent_sites": "v913_zeroD_sobol_0088403",
        "v913_paper_control01_0086420_persistent_sites": "v913_zeroD_sobol_0086420",
    }
