from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_INSTALLER_PATH = ROOT / "scripts" / "install_v913_four_class_paper_selection.py"
CANONICAL_INSTALLER_PATH = (
    ROOT / "scripts" / "install_v913_four_class_paper_selection_canonical.py"
)
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


BASE = load_module(BASE_INSTALLER_PATH, "install_v913_four_class_base_for_labels")
CANONICAL = load_module(
    CANONICAL_INSTALLER_PATH, "install_v913_four_class_canonical_for_labels"
)


def handoff_rows():
    template = BASE.read_csv(BASE_REGISTRY)[0]
    rows = []
    for material_class, candidate_id, option_key in (
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
    ):
        row = {
            "option_key": option_key,
            "candidate_id": candidate_id,
            "paper_material_class": material_class,
            "final_class_rank": 1,
            "selection_role": "paper primary",
            "oneD_strict_gate_passed": True,
            "oneD_selection_score": 0.5,
        }
        for field in BASE.ACTIVE_FIELDS:
            row[field] = template[field]
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


def test_canonical_installer_emits_four_distinct_material_classes():
    base_rows = BASE.read_csv(BASE_REGISTRY)
    handoff = handoff_rows()
    _, installed, selected = CANONICAL.install_rows(
        base_rows, handoff, manifest(handoff)
    )

    assert [row["material_class"] for row in installed] == [
        "peak",
        "DBTT",
        "weakT",
        "ceramic",
    ]
    assert [row["material_class_2d"] for row in selected] == [
        "peak",
        "DBTT",
        "weakT",
        "ceramic",
    ]
    assert [row["candidate_id"] for row in installed] == [
        "v913_zeroD_sobol_0242980",
        "v913_zeroD_sobol_0202500",
        "v913_zeroD_sobol_0001111",
        "v913_zeroD_sobol_0002222",
    ]
