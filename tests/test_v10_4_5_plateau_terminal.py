from __future__ import annotations

from types import SimpleNamespace

from arrhenius_fracture import sharp_front_v10_4_5_plasticity_plateau_audited as v1045


def _base_metrics():
    return {
        "terminal_basis": "collapsed_adaptive_substep_stagnation",
        "criteria": {
            "no_crack_event_in_window": True,
            "negligible_crack_extension": True,
            "plastic_accommodation_dominant": False,
            "elastic_storage_flat": False,
            "load_carrying_capacity_collapsed": True,
            "adaptive_substep_stagnation": True,
        },
        "criteria_pass": False,
        "plastic_work_fraction_window": 0.0,
        "elastic_storage_fraction_window": 1.0,
        "B_final": 0.1,
    }


def _window():
    return [
        {
            "J_positive": 4.0,
            "Uapp": 2.0,
            "Ftop": 3.0,
            "B": 0.1,
            "lambda_c": 0.01,
            "nominal_progress_end": 10.0,
        }
    ]


def test_severe_substep_plateau_uses_positive_cumulative_bulk_work(monkeypatch):
    monkeypatch.setattr(
        v1045,
        "_BASE_SUBSTEP_METRICS",
        lambda window, args, **kwargs: _base_metrics(),
    )

    metrics = v1045._plateau_substep_metrics(
        _window(),
        SimpleNamespace(),
        cumulative_Wp=2.0,
    )

    assert metrics is not None
    assert metrics["terminal_basis"] == v1045.TERMINAL_BASIS
    assert metrics["criteria_pass"] is True
    assert metrics["criteria"] == {
        "no_crack_event_in_window": True,
        "negligible_crack_extension": True,
        "bulk_plastic_dissipation_present": True,
        "load_carrying_response_plateau": True,
        "adaptive_substep_stagnation": True,
    }
    assert (
        metrics["v10_4_4_stagnation_criteria_diagnostics"]
        ["plastic_accommodation_dominant"]
        is False
    )
    assert metrics["incremental_plastic_fraction_role"] == (
        "diagnostic_only_for_severe_substep_fallback"
    )

    final = v1045._plateau_campaign_metrics(
        _window(),
        metrics,
        Eprime=25.0,
    )
    assert final["criteria_pass"] is True
    assert final["criteria"] == metrics["criteria"]
    assert final["terminal_classifier_model_id"] == v1045.MODEL_ID


def test_plateau_fallback_rejects_zero_cumulative_bulk_work(monkeypatch):
    monkeypatch.setattr(
        v1045,
        "_BASE_SUBSTEP_METRICS",
        lambda window, args, **kwargs: _base_metrics(),
    )

    metrics = v1045._plateau_substep_metrics(
        _window(),
        SimpleNamespace(),
        cumulative_Wp=0.0,
    )

    assert metrics is not None
    assert metrics["criteria_pass"] is False
    assert metrics["criteria"]["bulk_plastic_dissipation_present"] is False
