from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from arrhenius_fracture import (
    sharp_front_v10_4_7_numerical_stagnation_audited as v1047,
)


def _fallback(*, dominant: bool) -> dict:
    return {
        "terminal_basis": "collapsed_adaptive_substep_stagnation_v1046",
        "criteria_pass": dominant,
        "criteria": {
            "no_crack_event_in_window": True,
            "negligible_crack_extension": True,
            "cumulative_bulk_plastic_work_dominant": dominant,
            "load_carrying_response_plateau": True,
            "adaptive_substep_stagnation": True,
        },
        "classification_window_steps": 128,
        "window_first_step": 93,
        "window_last_step": 220,
        "cumulative_plastic_fraction_acceptance": (
            0.9487 if dominant else 1.68e-6
        ),
        "minimum_cumulative_plastic_fraction_acceptance": 0.90,
        "crack_extension_window_m": 0.0,
        "maximum_accepted_trial_fraction_window": 1.0e-8,
    }


def test_low_plastic_severe_plateau_is_numerical_stagnation():
    assert v1047._is_numerical_stagnation(_fallback(dominant=False)) is True


def test_physical_plastic_terminal_is_not_numerical_stagnation():
    assert v1047._is_numerical_stagnation(_fallback(dominant=True)) is False


def test_loader_raises_after_failed_physical_terminal(monkeypatch):
    module = SimpleNamespace(
        _v1042_terminal_metrics=lambda window, args, **kwargs: {
            "criteria_pass": False,
            "terminal_basis": "standard_nominal_window",
        }
    )
    monkeypatch.setattr(
        v1047._v1046,
        "_load_transformed_sharp_front_v1046",
        lambda: module,
    )
    monkeypatch.setattr(
        v1047._v1046,
        "_dominance_substep_metrics",
        lambda window, args, **kwargs: _fallback(dominant=False),
    )

    transformed = v1047._load_transformed_sharp_front_v1047()
    with pytest.raises(v1047.NumericalStagnationError) as caught:
        transformed._v1042_terminal_metrics(
            [],
            SimpleNamespace(),
            Eprime=25.0,
        )

    assert (
        caught.value.metrics["cumulative_plastic_fraction_acceptance"]
        == 1.68e-6
    )


def test_passing_physical_terminal_retains_priority(monkeypatch):
    physical = {
        "criteria_pass": True,
        "terminal_basis": "collapsed_adaptive_substep_stagnation_v1046",
    }
    module = SimpleNamespace(
        _v1042_terminal_metrics=lambda window, args, **kwargs: physical
    )
    monkeypatch.setattr(
        v1047._v1046,
        "_load_transformed_sharp_front_v1046",
        lambda: module,
    )
    monkeypatch.setattr(
        v1047._v1046,
        "_dominance_substep_metrics",
        lambda *args, **kwargs: pytest.fail(
            "stagnation arbitration must not run after a passing physical terminal"
        ),
    )

    transformed = v1047._load_transformed_sharp_front_v1047()
    assert transformed._v1042_terminal_metrics([], SimpleNamespace()) is physical


def test_writer_is_fail_closed(tmp_path):
    for name in ("COMPLETE", "PLASTIC_FLOW", "PLASTICITY_DOMINATED"):
        (tmp_path / name).write_text("stale\n")

    v1047._write_numerical_stagnation(
        tmp_path,
        _fallback(dominant=False),
    )

    assert (tmp_path / "NUMERICAL_STAGNATION").is_file()
    assert (tmp_path / "numerical_stagnation_audit.json").is_file()
    for name in ("COMPLETE", "PLASTIC_FLOW", "PLASTICITY_DOMINATED"):
        assert not (tmp_path / name).exists()

    audit = json.loads(
        (tmp_path / "numerical_stagnation_audit.json").read_text()
    )
    assert audit["complete"] is False
    assert audit["plasticity_dominated"] is False
    assert audit["physical_plasticity_terminal_accepted"] is False
    assert audit["exit_code"] == 4
    assert audit["classification"] == (
        "numerical_stagnation_not_plasticity_dominated"
    )


def test_main_writes_audit_and_exits_nonzero(monkeypatch, tmp_path):
    module = SimpleNamespace(
        _v1042_terminal_metrics=lambda window, args, **kwargs: {
            "criteria_pass": False,
            "terminal_basis": "standard_nominal_window",
        }
    )
    monkeypatch.setattr(
        v1047._v1046,
        "_load_transformed_sharp_front_v1046",
        lambda: module,
    )
    monkeypatch.setattr(
        v1047._v1046,
        "_dominance_substep_metrics",
        lambda window, args, **kwargs: _fallback(dominant=False),
    )

    def fake_v1044_main(args):
        transformed = v1047._v1044.load_transformed_sharp_front()
        transformed._v1042_terminal_metrics([], SimpleNamespace())
        pytest.fail("numerical stagnation did not abort the solver")

    monkeypatch.setattr(v1047._v1044, "main", fake_v1044_main)

    with pytest.raises(SystemExit) as caught:
        v1047.main(["--out", str(tmp_path)])

    assert caught.value.code == v1047.NUMERICAL_STAGNATION_EXIT_CODE
    assert (tmp_path / "NUMERICAL_STAGNATION").is_file()
    audit = json.loads(
        (tmp_path / "numerical_stagnation_audit.json").read_text()
    )
    assert audit["window_first_step"] == 93
    assert audit["window_last_step"] == 220
