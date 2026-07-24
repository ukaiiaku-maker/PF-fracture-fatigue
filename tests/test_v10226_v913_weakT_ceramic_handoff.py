from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

from arrhenius_fracture.sharp_front_v10_2_26 import VALID_OPTIONS


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "install_v913_weakT_ceramic_handoff.py"
SPEC = importlib.util.spec_from_file_location("install_v913_weakT_ceramic_handoff", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _active(seed: float) -> dict[str, float]:
    values = {field: seed + 0.01 * index for index, field in enumerate(MODULE.ACTIVE_FIELDS)}
    values["Tref_K"] = 481.33
    values["peierls_nu0_s"] = 1.0e12
    values["taylor_nu0_s"] = 1.0e11
    return values


def _handoff_rows() -> list[dict[str, object]]:
    rows = []
    for index, (material_class, expected) in enumerate(MODULE.EXPECTED.items(), start=1):
        rows.append(
            {
                "option_key": expected["option_key"],
                "candidate_id": expected["candidate_id"],
                "paper_material_class": material_class,
                "selection_role": expected["role"],
                "oneD_strict_gate_passed": material_class == "ceramic_like",
                "oneD_selection_score": float(index),
                **_active(float(index)),
            }
        )
    return rows


def _manifest(rows: list[dict[str, object]]) -> dict[str, object]:
    selected = []
    for row in rows:
        selected.append(
            {
                "option_key": row["option_key"],
                "candidate_id": row["candidate_id"],
                "paper_material_class": row["paper_material_class"],
                "selection_role": row["selection_role"],
                "oneD_strict_gate_passed": row["oneD_strict_gate_passed"],
                "oneD_selection_score": row["oneD_selection_score"],
                "oneD_metrics": {
                    "K50_temperature_span_MPa_sqrt_m": 1.0,
                    "high_temperature_toughness_loss_MPa_sqrt_m": 1.0,
                    "median_R_rise_first_to_50_MPa_sqrt_m": 1.0,
                    "median_R_rise_25_to_50_MPa_sqrt_m": 0.0,
                },
            }
        )
    return {
        "schema": "v9.13_weakT_ceramic_paper_handoff_v1",
        "selected": selected,
        "fixed_closure": {
            "persistent_sites": True,
            "finite_source_inventory": False,
            "source_refresh_on_crack_advance": False,
            "explicit_recovery": False,
        },
        "transfer_policy": "exact",
    }


def _read_template() -> list[dict[str, str]]:
    with MODULE.DEFAULT_TEMPLATE.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_installer_preserves_active_rows_and_fixed_contract() -> None:
    handoff = _handoff_rows()
    manifest = _manifest(handoff)
    fields, target, selected = MODULE.build_registry(handoff, manifest, _read_template())
    assert len(target) == 2
    assert len(selected) == 2
    assert fields == list(_read_template()[0])

    by_id = {row["candidate_id"]: row for row in target}
    for source in handoff:
        row = by_id[source["candidate_id"]]
        for field in MODULE.ACTIVE_FIELDS:
            assert float(row[field]) == float(source[field])
        assert float(row["source_refresh_length_um"]) == 0.0
        assert float(row["explicit_recovery_active"]) == 0.0


def test_active_fingerprint_is_order_independent() -> None:
    rows = _handoff_rows()
    assert MODULE.active_fingerprint(rows) == MODULE.active_fingerprint(list(reversed(rows)))


def test_v10226_option_mapping_is_stable() -> None:
    assert VALID_OPTIONS == {
        "v913_paper_weakT01_0257068_persistent_sites": "v913_zeroD_sobol_0257068",
        "v913_paper_ceramic01_0189364_persistent_sites": "v913_zeroD_sobol_0189364",
    }
