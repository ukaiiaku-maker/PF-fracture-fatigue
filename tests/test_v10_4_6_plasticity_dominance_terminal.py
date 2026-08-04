from __future__ import annotations

from types import SimpleNamespace

from arrhenius_fracture import (
    sharp_front_v10_4_6_plasticity_dominance_audited as v1046,
)


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


def _base_metrics(cumulative_fraction: float):
    return {
        "terminal_basis": "collapsed_adaptive_substep_stagnation",
        "criteria": {
            "no_crack_event_in_window": True,
            "negligible_crack_extension": True,
            "plastic_accommodation_dominant": cumulative_fraction >= 0.90,
            "elastic_storage_flat": False,
            "load_carrying_capacity_collapsed": True,
            "adaptive_substep_stagnation": True,
        },
        "criteria_pass": False,
        "plastic_work_fraction_window": 0.0,
        "elastic_storage_fraction_window": 1.0,
        "cumulative_plastic_fraction": cumulative_fraction,
        "B_final": 0.1,
    }


def test_peak_like_tiny_positive_plastic_work_is_rejected(monkeypatch):
    monkeypatch.setattr(
        v1046,
        "_BASE_SUBSTEP_METRICS",
        lambda window, args, **kwargs: _base_metrics(1.68e-6),
    )

    metrics = v1046._dominance_substep_metrics(
        _window(),
        SimpleNamespace(
            plastic_flow_min_cumulative_plastic_fraction=0.90,
        ),
    )

    assert metrics is not None
    assert metrics["criteria_pass"] is False
    assert metrics["criteria"]["cumulative_bulk_plastic_work_dominant"] is False
    assert metrics["cumulative_plastic_fraction_acceptance"] == 1.68e-6
    assert metrics["minimum_cumulative_plastic_fraction_acceptance"] == 0.90


def test_dbtt_like_cumulative_plastic_dominance_is_accepted(monkeypatch):
    monkeypatch.setattr(
        v1046,
        "_BASE_SUBSTEP_METRICS",
        lambda window, args, **kwargs: _base_metrics(0.9487),
    )

    metrics = v1046._dominance_substep_metrics(
        _window(),
        SimpleNamespace(
            plastic_flow_min_cumulative_plastic_fraction=0.90,
        ),
    )

    assert metrics is not None
    assert metrics["criteria_pass"] is True
    assert metrics["criteria"] == {
        "no_crack_event_in_window": True,
        "negligible_crack_extension": True,
        "cumulative_bulk_plastic_work_dominant": True,
        "load_carrying_response_plateau": True,
        "adaptive_substep_stagnation": True,
    }

    final = v1046._dominance_campaign_metrics(
        _window(),
        metrics,
        Eprime=25.0,
    )
    assert final["criteria_pass"] is True
    assert final["criteria"] == metrics["criteria"]
    assert final["terminal_classifier_model_id"] == v1046.MODEL_ID


def test_failed_primary_does_not_override_failed_physical_fallback(monkeypatch):
    primary = {
        "criteria_pass": False,
        "terminal_basis": "standard_nominal_window",
    }
    module = SimpleNamespace(
        _v1042_terminal_metrics=lambda window, args, **kwargs: primary
    )
    monkeypatch.setattr(
        v1046._terminal,
        "load_transformed_sharp_front",
        lambda: module,
    )
    monkeypatch.setattr(
        v1046,
        "_dominance_substep_metrics",
        lambda window, args, **kwargs: {
            "criteria_pass": False,
            "terminal_basis": v1046.TERMINAL_BASIS,
        },
    )

    transformed = v1046._load_transformed_sharp_front_v1046()
    result = transformed._v1042_terminal_metrics(
        _window(),
        SimpleNamespace(),
        Eprime=25.0,
    )

    assert result is primary


def test_failed_primary_allows_passing_physical_fallback(monkeypatch):
    module = SimpleNamespace(
        _v1042_terminal_metrics=lambda window, args, **kwargs: {
            "criteria_pass": False,
            "terminal_basis": "standard_nominal_window",
        }
    )
    monkeypatch.setattr(
        v1046._terminal,
        "load_transformed_sharp_front",
        lambda: module,
    )
    monkeypatch.setattr(
        v1046,
        "_dominance_substep_metrics",
        lambda window, args, **kwargs: {
            "criteria_pass": True,
            "terminal_basis": v1046.TERMINAL_BASIS,
        },
    )
    monkeypatch.setattr(
        v1046,
        "_dominance_campaign_metrics",
        lambda window, metrics, Eprime: {
            "criteria_pass": True,
            "terminal_basis": metrics["terminal_basis"],
            "selected": "physical_dominance_fallback",
        },
    )

    transformed = v1046._load_transformed_sharp_front_v1046()
    result = transformed._v1042_terminal_metrics(
        _window(),
        SimpleNamespace(),
        Eprime=25.0,
    )

    assert result["criteria_pass"] is True
    assert result["selected"] == "physical_dominance_fallback"
