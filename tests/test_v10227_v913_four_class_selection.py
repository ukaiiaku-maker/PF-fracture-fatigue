from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
INSTALLER_PATH = ROOT / "scripts" / "install_v913_four_class_paper_selection.py"
BASE_REGISTRY = (
    ROOT
    / "arrhenius_fracture"
    / "data"
    / "materials"
    / "v10_2_25_v913_paper_campaign_registry.csv"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


INSTALLER = load_module(INSTALLER_PATH, "install_v913_four_class")


def handoff_rows():
    base = INSTALLER.read_csv(BASE_REGISTRY)[0]
    rows = []
    specs = [
        (
            "weakT_FCC_like",
            "v913_zeroD_sobol_0001111",
            "v913_paper_weakT01_0001111_persistent_sites",
        ),
        (
            "ceramic_like",
            "v913_zeroD_sobol_0002222",
            "v913_paper_ceramic01_0002222_persistent_sites",
        ),
    ]
    for material_class, candidate_id, option_key in specs:
        row = {
            "option_key": option_key,
            "candidate_id": candidate_id,
            "paper_material_class": material_class,
            "final_class_rank": 1,
            "selection_role": "paper primary",
            "oneD_strict_gate_passed": True,
            "oneD_selection_score": 0.5,
        }
        for field in INSTALLER.ACTIVE_FIELDS:
            row[field] = base[field]
        rows.append(row)
    return rows


def manifest(rows):
    return {
        "schema": "v9.13_weakT_ceramic_100um_final_selection_v1",
        "primary_candidates": [
            {
                "paper_material_class": row["paper_material_class"],
                "candidate_id": row["candidate_id"],
                "option_key": row["option_key"],
                "final_class_rank": 1,
                "selection_role": "paper primary",
                "oneD_strict_gate_passed": True,
                "oneD_selection_score": 0.5,
                "oneD_metrics": {},
            }
            for row in rows
        ],
        "backup_candidates": [],
        "fixed_closure": {
            "persistent_sites": True,
            "finite_source_inventory": False,
            "source_refresh_on_crack_advance": False,
            "explicit_recovery": False,
        },
        "transfer_policy": "exact active-row transfer only",
    }


def test_installs_exactly_four_primary_classes():
    base_rows = INSTALLER.read_csv(BASE_REGISTRY)
    handoff = handoff_rows()
    fields, installed, selected = INSTALLER.install_rows(
        base_rows, handoff, manifest(handoff)
    )

    assert len(fields) == len(base_rows[0])
    assert [row["candidate_id"] for row in installed] == [
        "v913_zeroD_sobol_0242980",
        "v913_zeroD_sobol_0202500",
        "v913_zeroD_sobol_0001111",
        "v913_zeroD_sobol_0002222",
    ]
    assert [row["option_key"] for row in installed] == [
        "v913_paper_peak01_0242980_persistent_sites",
        "v913_paper_dbtt01_0202500_persistent_sites",
        "v913_paper_weakT01_0001111_persistent_sites",
        "v913_paper_ceramic01_0002222_persistent_sites",
    ]
    assert len(selected) == 4
    assert all(float(row["source_refresh_length_um"]) == 0.0 for row in installed)
    assert all(float(row["explicit_recovery_active"]) == 0.0 for row in installed)


def test_rejects_non_strict_primary():
    base_rows = INSTALLER.read_csv(BASE_REGISTRY)
    handoff = handoff_rows()
    handoff[1]["oneD_strict_gate_passed"] = False
    with pytest.raises(RuntimeError, match="did not pass the strict"):
        INSTALLER.install_rows(base_rows, handoff, manifest(handoff))
