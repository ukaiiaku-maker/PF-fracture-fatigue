from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from arrhenius_fracture.sharp_front_v10_1_7_5 import (
    _env_nonnegative,
    _require_transfer_scope,
)
from scripts.analyze_v10_1_7_5_reduced_candidate_transfer import _candidate_summary
from scripts.prepare_v10_1_7_5_candidate_transfer import (
    MODES,
    REQUIRED_FIELDS,
    prepare,
)


def _candidate_row(candidate_id: str, temperatures: list[float], c_blunt: float = 2.0):
    row = {name: 1.0 for name in REQUIRED_FIELDS}
    row.update(
        {
            "candidate_id": candidate_id,
            "transition_bracket": f"T{temperatures[0]:04.0f}_{temperatures[-1]:04.0f}K",
            "refinement_transition_temperatures_K": json.dumps(temperatures),
            "c_blunt": c_blunt,
        }
    )
    return row


def test_preparation_builds_two_endpoints_six_modes_and_true_blunting_ablation(tmp_path: Path):
    source = pd.DataFrame(
        [
            _candidate_row("DBTT_A0003408", [500.0, 533.3, 566.7, 600.0]),
            _candidate_row("DBTT_A0000353", [700.0, 733.3, 766.7, 800.0]),
        ]
    )
    cases = prepare(source, ["DBTT_A0003408", "DBTT_A0000353"], tmp_path)
    assert len(cases) == 24
    assert set(cases["mode"]) == set(MODES)
    assert set(cases.groupby("candidate_id").T_K.unique().explode()) == {
        500.0,
        600.0,
        700.0,
        800.0,
    }
    assert set(cases.forest_density_floor_override_m2.astype(str)) == {
        "default",
        "0.0",
    }

    full = pd.read_csv(
        tmp_path / "material_manifests" / "DBTT_A0003408" / "candidate.csv"
    )
    blunt = pd.read_csv(
        tmp_path
        / "material_manifests"
        / "DBTT_A0003408"
        / "candidate_blunting_off.csv"
    )
    assert float(full.c_blunt.iloc[0]) == 2.0
    assert float(blunt.c_blunt.iloc[0]) == 0.0


def test_transfer_scope_requires_deterministic_single_front_tip_only(monkeypatch):
    monkeypatch.setenv("ANISOTROPIC_USE_AVALANCHE_BACKEND", "0")
    monkeypatch.setenv("CLEAVAGE_HAZARD_MODE", "deterministic")
    monkeypatch.setenv("CLEAVAGE_EVENT_LENGTH_MODE", "fixed")
    args = [
        "--bulk-plasticity-mode",
        "tip_only",
        "--max-fronts",
        "1",
        "--no-wake-shielding",
    ]
    _require_transfer_scope(args)

    with pytest.raises(SystemExit):
        _require_transfer_scope(
            [
                "--bulk-plasticity-mode",
                "full_field",
                "--max-fronts",
                "1",
                "--no-wake-shielding",
            ]
        )


def test_zero_backstress_and_zero_forest_overrides_are_allowed(monkeypatch):
    monkeypatch.setenv("V10175_BACKSTRESS_SCALE", "0")
    monkeypatch.setenv("V10175_FOREST_DENSITY_FLOOR_M2", "0")
    assert _env_nonnegative("V10175_BACKSTRESS_SCALE", 1.0) == 0.0
    assert _env_nonnegative("V10175_FOREST_DENSITY_FLOOR_M2", None) == 0.0


def test_transfer_scoring_prefers_emission_blunting_without_background_or_shielding():
    rows = []
    values = {
        "full": (10.0, 20.0, 1.0, 100.0),
        "plasticity_off": (10.0, 9.5, 0.0, 0.0),
        "blunting_off": (10.0, 10.0, 0.0, 0.0),
        "backstress_off": (10.0, 22.0, 1.0, 120.0),
        "shielding_off": (10.0, 19.5, 0.0, 100.0),
        "background_field_off": (10.0, 19.0, 1.0, 95.0),
    }
    for mode, (low, high, shield, emission_growth) in values.items():
        rows.append(
            {
                "candidate_id": "candidate",
                "transition_bracket": "T0700_0800K",
                "mode": mode,
                "low_K_init_MPa_sqrt_m": low,
                "high_K_init_MPa_sqrt_m": high,
                "rise_MPa_sqrt_m": high - low,
                "endpoint_ratio": high / low,
                "emission_budget_growth": emission_growth,
                "max_abs_K_shield_MPa_sqrt_m": shield,
                "max_abs_sigma_back_channel_Pa": 1.0e9,
                "minimum_tensor_reliable_fraction": 1.0,
                "post_hazard_weighting_count": 0,
            }
        )
    result = _candidate_summary("candidate", pd.DataFrame(rows))
    assert result["two_d_transfer_priority"] is True
    assert result["emission_fraction_of_full_rise"] > 0.60
    assert result["blunting_sensitivity_fraction"] >= 0.50
    assert abs(result["shielding_sensitivity_fraction"]) <= 0.20
    assert result["background_off_retained_rise_fraction"] >= 0.75
