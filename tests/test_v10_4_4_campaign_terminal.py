from __future__ import annotations

import json
import math
from types import SimpleNamespace

from arrhenius_fracture import plastic_flow_terminal_v1042 as v1042
from arrhenius_fracture.plastic_flow_campaign_terminal_v1044 import (
    AUDIT_SCHEMA,
    MODEL_ID,
    _campaign_metrics,
    _campaign_terminal_block,
    _substep_stagnation_metrics,
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


def _stagnation_args(**updates):
    values = {
        "plastic_flow_stagnation_substeps": 128,
        "plastic_flow_stagnation_max_trial_fraction": 1.0e-6,
        "plastic_flow_max_da_fraction": 0.1,
        "plastic_flow_min_plastic_fraction": 0.90,
        "plastic_flow_min_cumulative_plastic_fraction": 0.90,
        "plastic_flow_max_elastic_fraction": 0.05,
        "plastic_flow_max_force_fraction": 0.10,
        "plastic_flow_max_tangent_fraction": 0.05,
        "plastic_flow_stagnation_plateau_rel_tol": 1.0e-3,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _stagnation_window(*, trial_fraction=1.0e-8):
    rows = []
    for index in range(128):
        start = index * trial_fraction
        end = (index + 1) * trial_fraction
        rows.append(
            {
                "step": index + 1,
                "nominal_progress_start": start,
                "nominal_progress_end": end,
                "Uapp": index * 1.0e-12,
                "Ftop": 100.0,
                "J_positive": 10.0,
                "sigma_tip": 1.0,
                "B": index * 1.0e-12,
                "lambda_c": 1.0e-12,
                "n_fire": 0,
                "a_tip": 5.0e-4,
                "W_ext": float(index),
                "U_el": 0.01 * index,
                "W_p": 0.99 * index,
                "W_emit": 0.0,
            }
        )
    return rows


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
    window = [
        {
            "J_positive": 0.0,
            "Uapp": 0.0,
            "Ftop": 0.0,
            "B": 0.0,
            "lambda_c": 0.0,
        }
    ]
    result = _campaign_metrics(window, metrics, Eprime=1.0)
    assert result["criteria_pass"] is False
    assert result["criteria"]["plastic_accommodation_dominant"] is False


def test_bounded_substep_stagnation_terminal_accepts_plastic_plateau():
    window = _stagnation_window()
    metrics = _substep_stagnation_metrics(
        window,
        _stagnation_args(),
        da_phys=5.0e-6,
        peak_force=100.0,
        stiffness_reference=1.0e15,
        cumulative_Wp=100.0,
        cumulative_Uel=1.0,
        cumulative_Wemit=0.0,
    )
    assert metrics is not None
    assert metrics["terminal_basis"] == "collapsed_adaptive_substep_stagnation"
    assert metrics["criteria_pass"] is True
    assert metrics["criteria"]["adaptive_substep_stagnation"] is True
    assert math.isclose(
        metrics["maximum_accepted_trial_fraction_window"],
        1.0e-8,
        rel_tol=1.0e-10,
    )
    result = _campaign_metrics(window, metrics, Eprime=25.0)
    assert result["criteria_pass"] is True
    assert "adaptive_substep_stagnation" in result["criteria"]


def test_bounded_substep_stagnation_rejects_normal_sized_steps():
    window = _stagnation_window(trial_fraction=1.0e-3)
    metrics = _substep_stagnation_metrics(
        window,
        _stagnation_args(),
        da_phys=5.0e-6,
        peak_force=100.0,
        stiffness_reference=1.0e15,
        cumulative_Wp=100.0,
        cumulative_Uel=1.0,
        cumulative_Wemit=0.0,
    )
    assert metrics is not None
    assert metrics["criteria_pass"] is False
    assert metrics["criteria"]["adaptive_substep_stagnation"] is False


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
