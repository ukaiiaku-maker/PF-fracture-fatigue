from __future__ import annotations

import csv
from pathlib import Path

import pytest

from arrhenius_fracture.prospective_campaign import REGISTRY, TEMPERATURES_K
from arrhenius_fracture.taylor_factorization import (
    HISTORICAL_JUMP_FLOOR_M,
    HISTORICAL_PHI_CAP,
    evaluate_same_state,
    raw_domain_status,
    select_temperature_anchors,
)

ROOT = Path(__file__).resolve().parents[1]


def _rows():
    return list(csv.DictReader((ROOT / REGISTRY).open(newline="")))


def test_anchor_selection_never_evaluates_unsupported_raw_parent_state():
    for row in _rows():
        anchors, domain = select_temperature_anchors(row, tuple(TEMPERATURES_K))
        assert 1 <= len(anchors) <= 3
        assert all(raw_domain_status(row, T)["status"] == "ADMISSIBLE" for T in anchors)
        if row["material_class"] in {"peak", "DBTT"}:
            assert any(item["status"] == "UNSUPPORTED_RAW_SCALE" for item in domain)


def test_factorization_isolates_each_switch_and_preserves_signed_log_policy():
    row = next(item for item in _rows() if item["material_class"] == "DBTT")
    rho_c = float(row["taylor_corr_rho_c_m2"])
    result = evaluate_same_state(
        row, temperature_K=600.0, signed_local_stress_Pa=5.0e8,
        raw_density_m2=100.0 * rho_c,
    )
    variants = {item["name"]: item for item in result["isolated_variants"]}
    current = variants["current_combined"]
    assert variants["shifted_density_order_only"]["order"] != current["order"]
    assert result["geometry"]["phi_uncapped"] > HISTORICAL_PHI_CAP
    assert variants["phi_cap_20_only"]["taylor_net_rate_s"] != current["taylor_net_rate_s"]
    assert set(result["current_elementary"]["taylor"]) >= {
        "log_forward_rate", "log_reverse_rate", "signed_net_rate_s"
    }
    assert "log_net_rate" not in result["current_elementary"]["taylor"]


def test_zero_bias_cancels_current_but_exact_old_tip_remains_one_way():
    row = _rows()[0]
    result = evaluate_same_state(
        row, temperature_K=600.0, signed_local_stress_Pa=0.0,
        raw_density_m2=float(row["taylor_corr_rho_c_m2"]),
    )
    assert result["current_elementary"]["peierls"]["signed_net_rate_s"] == 0.0
    assert result["current_elementary"]["taylor"]["signed_net_rate_s"] == 0.0
    assert result["historical_moving_tip_exact"]["peierls_one_way_rate_s"] > 0.0


def test_raw_and_consumed_density_and_jump_floor_are_both_reported():
    row = _rows()[0]
    result = evaluate_same_state(
        row, temperature_K=600.0, signed_local_stress_Pa=-5.0e8,
        raw_density_m2=1.0,
    )
    assert result["density_floor_was_active"] is True
    assert result["consumed_forest_density_m2"] == float(row["rho_forest_floor_m2"])
    jump_variant = next(
        item for item in result["isolated_variants"]
        if item["name"] == "jump_floor_2p5e-10_m_only"
    )
    assert jump_variant["jump_length_m"] >= HISTORICAL_JUMP_FLOOR_M


def test_unsupported_state_returns_raw_values_without_rates():
    row = next(item for item in _rows() if item["material_class"] == "peak")
    result = evaluate_same_state(
        row, temperature_K=1300.0, signed_local_stress_Pa=1.0e8,
        raw_density_m2=float(row["taylor_corr_rho_c_m2"]),
    )
    assert result["status"] == "NOT_EVALUATED_UNSUPPORTED_RAW_SCALE"
    assert result["domain"]["invalid_raw_scales"]
    assert "isolated_variants" not in result


def test_rate_cap_and_jump_floor_are_inactive_when_not_reached():
    row = _rows()[-1]
    result = evaluate_same_state(
        row, temperature_K=950.0, signed_local_stress_Pa=2.0e8,
        raw_density_m2=float(row["taylor_corr_rho_c_m2"]),
    )
    variants = {item["name"]: item for item in result["isolated_variants"]}
    capped = variants["elementary_rate_cap_1e12_only"]
    assert capped["rate_after_optional_cap_s"] <= 1.0e12
    assert capped["completion_rate_s"] == pytest.approx(
        variants["current_combined"]["completion_rate_s"]
    )
