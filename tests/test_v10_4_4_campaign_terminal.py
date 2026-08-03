from __future__ import annotations

import json
import math

from arrhenius_fracture import plastic_flow_terminal_v1042 as v1042
from arrhenius_fracture.plastic_flow_campaign_terminal_v1044 import (
    AUDIT_SCHEMA,
    MODEL_ID,
    _campaign_metrics,
    _campaign_terminal_block,
)
from arrhenius_fracture.sharp_front_v10_4_4_plasticity_dominated_audited import (
    _rewrite_terminal_outputs,
)


def _criteria(**updates):
    values = {
        "no_crack_event_in_window": True,
        "negligible_crack_extension": True,
        "negligible_positive_tip_J": False,
        "negligible_tip_stress": False,
        "plastic_accommodation_dominant": True,
        "elastic_storage_flat": True,
        "load_carrying_capacity_collapsed": True,
        "cleavage_clock_stalled": False,
        "projected_cleavage_action_safe": False,
    }
    values.update(updates)
    return values


def test_terminal_block_allows_post_first_passage_and_writes_new_marker():
    patched = _campaign_terminal_block(v1042._TERMINAL_BLOCK)
    assert "and not fatigue_mode and Kc_first is None" not in patched
    assert "and not fatigue_mode):" in patched
    assert "plasticity_dominated_after_partial_fracture" in patched
    assert "plasticity_dominated_no_crack_growth" in patched
    assert "PLASTICITY_DOMINATED" in patched
    assert "'sharp_fracture_occurred': bool(Kc_first is not None)" in patched
    assert "'Eprime_Pa': float(mat.Eprime)" in patched


def test_campaign_acceptance_ignores_future_hazard_and_finite_tip_fields():
    metrics = {
        "criteria": _criteria(),
        "B_final": 0.75,
        "projected_cleavage_action_increment": 100.0,
        "projected_hazard_fraction_of_remaining_budget": 400.0,
    }
    window = [
        {
            "J_positive": 4.0,
            "Uapp": 2.0,
            "Ftop": 3.0,
            "B": 0.75,
            "lambda_c": 8.0,
            "nominal_progress_end": 50.0,
        }
    ]
    result = _campaign_metrics(window, metrics, Eprime=25.0)
    assert result["schema"] == AUDIT_SCHEMA
    assert result["terminal_classifier_model_id"] == MODEL_ID
    assert result["criteria_pass"] is True
    assert set(result["criteria"]) == {
        "no_crack_event_in_window",
        "negligible_crack_extension",
        "plastic_accommodation_dominant",
        "elastic_storage_flat",
        "load_carrying_capacity_collapsed",
    }
    assert result["projected_cleavage_action_is_diagnostic_only"] is True
    assert result["cleavage_action_growth_is_diagnostic_only"] is True
    assert result["J_elastic_positive_terminal_J_per_m2"] == 4.0
    assert math.isclose(
        result["K_elastic_equivalent_terminal_MPa_sqrt_m"],
        10.0e-6,
    )


def test_campaign_terminal_still_requires_plasticity_dominance():
    metrics = {"criteria": _criteria(plastic_accommodation_dominant=False)}
    window = [{"J_positive": 0.0, "Uapp": 0.0, "Ftop": 0.0, "B": 0.0, "lambda_c": 0.0}]
    result = _campaign_metrics(window, metrics, Eprime=1.0)
    assert result["criteria_pass"] is False
    assert result["criteria"]["plastic_accommodation_dominant"] is False


def test_terminal_output_reports_apparent_plasticity_limited_toughness(tmp_path):
    audit = {
        "classification": "plasticity_dominated_after_partial_fracture",
        "J_tip_positive_final_J_per_m2": 4.0,
        "J_pl_diss_J_per_m2": 5.0,
        "Eprime_Pa": 100.0,
    }
    (tmp_path / "plastic_flow_terminal_audit.json").write_text(
        json.dumps(audit) + "\n"
    )
    (tmp_path / "summary.json").write_text(
        json.dumps([{"campaign_terminal": True, "mode": "plastic-flow"}]) + "\n"
    )

    _rewrite_terminal_outputs(tmp_path)

    rewritten = json.loads(
        (tmp_path / "plastic_flow_terminal_audit.json").read_text()
    )
    assert rewritten["J_elastic_positive_J_per_m2"] == 4.0
    assert rewritten["J_plastic_dissipation_J_per_m2"] == 5.0
    assert rewritten["J_apparent_total_J_per_m2"] == 9.0
    assert math.isclose(
        rewritten["K_apparent_plasticity_limited_MPa_sqrt_m"],
        30.0e-6,
    )
    assert rewritten["apparent_toughness_label"].endswith("not_K_IC")
    assert (
        tmp_path / "PLASTICITY_DOMINATED"
    ).read_text().strip() == "plasticity_dominated_after_partial_fracture"

    summary = json.loads((tmp_path / "summary.json").read_text())[0]
    assert summary["terminal_status"] == "plasticity_dominated_after_partial_fracture"
    assert summary["J_apparent_total_J_per_m2"] == 9.0
