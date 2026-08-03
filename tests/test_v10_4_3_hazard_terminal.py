from __future__ import annotations

from collections import deque
from pathlib import Path
from types import SimpleNamespace

from arrhenius_fracture.plastic_flow_hazard_terminal_v1043 import (
    load_transformed_sharp_front,
    transform_source,
)


def _source() -> str:
    return Path("arrhenius_fracture/sharp_front.py").read_text()


def _args(**overrides):
    values = {
        "plastic_flow_window_steps": 3,
        "plastic_flow_min_step": 3,
        "plastic_flow_max_da_fraction": 0.1,
        "plastic_flow_J_rel_tol": 1.0e-6,
        "plastic_flow_J_abs_tol_J_per_m2": 1.0e-6,
        "plastic_flow_sigma_rel_tol": 1.0e-6,
        "plastic_flow_min_plastic_fraction": 0.90,
        "plastic_flow_min_cumulative_plastic_fraction": 0.90,
        "plastic_flow_max_elastic_fraction": 0.05,
        "plastic_flow_max_force_fraction": 0.10,
        "plastic_flow_max_tangent_fraction": 0.05,
        "plastic_flow_max_dB_window": 1.0e-6,
        "plastic_flow_min_cleavage_horizon_ratio": 100.0,
        "plastic_flow_prospective_horizon_steps": 2000.0,
        "plastic_flow_max_projected_hazard_fraction": 0.01,
        "plastic_flow_hazard_growth_percentile": 95.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _row(
    index: int,
    *,
    force: float,
    wext: float,
    uel: float,
    wp: float,
    lambda_c: float,
    J_positive: float = 5.0e-3,
    sigma_tip: float = 2.0e7,
):
    start = float(index - 1)
    end = float(index)
    return {
        "step": index,
        "nominal_progress_start": start,
        "nominal_progress_end": end,
        "Uapp": end * 1.0e-8,
        "Ftop": force,
        "J_positive": J_positive,
        "J_signed": J_positive,
        "sigma_tip": sigma_tip,
        "B": 0.0,
        "lambda_c": lambda_c,
        "n_fire": 0,
        "a_tip": 0.0,
        "W_ext": wext,
        "U_el": uel,
        "W_p": wp,
        "W_emit": 0.0,
    }


def _metrics(rows, args=None):
    module = load_transformed_sharp_front()
    args = _args() if args is None else args
    last = rows[-1]
    return module._v1042_terminal_metrics(
        deque(rows),
        args,
        Eprime=4.0e11,
        da_phys=1.0e-6,
        sigma_reference=1.0e9,
        peak_J_positive=max(row["J_positive"] for row in rows),
        peak_force=max(abs(row["Ftop"]) for row in rows),
        stiffness_reference=1.0e11,
        remaining_steps=0.0,
        nominal_dt_s=0.0328125,
        cumulative_Wp=last["W_p"],
        cumulative_Uel=last["U_el"],
        cumulative_Wemit=last["W_emit"],
    )


def test_hazard_terminal_transform_compiles_and_exposes_options() -> None:
    transformed = transform_source(_source())
    compile(transformed, "sharp_front.py[v10.4.3-hazard-terminal-test]", "exec")
    assert "--plastic-flow-prospective-horizon-steps" in transformed
    assert "--plastic-flow-max-projected-hazard-fraction" in transformed
    assert "projected_cleavage_action_safe" in transformed
    assert "J_and_sigma_zero_gates_are_diagnostic_only" in transformed
    assert "'negligible_positive_tip_J':" not in transformed
    assert "'negligible_tip_stress':" not in transformed


def test_nearly_elastic_load_bearing_state_fails() -> None:
    rows = [
        _row(i, force=1000.0 * i, wext=float(i), uel=0.95 * i,
             wp=0.05 * i, lambda_c=1.0e-80)
        for i in range(1, 5)
    ]
    metrics = _metrics(rows)
    assert metrics is not None
    assert not metrics["criteria_pass"]
    assert not metrics["criteria"]["plastic_accommodation_dominant"]
    assert not metrics["criteria"]["load_carrying_capacity_collapsed"]


def test_plastically_active_but_load_bearing_state_fails() -> None:
    rows = [
        _row(i, force=1000.0 * i, wext=float(i), uel=0.1,
             wp=0.95 * i, lambda_c=1.0e-80)
        for i in range(1, 5)
    ]
    metrics = _metrics(rows)
    assert metrics is not None
    assert not metrics["criteria_pass"]
    assert metrics["criteria"]["plastic_accommodation_dominant"]
    assert not metrics["criteria"]["load_carrying_capacity_collapsed"]


def test_collapsed_stiffness_inaccessible_hazard_passes_with_finite_J_and_stress() -> None:
    rows = [
        _row(i, force=4000.0, wext=float(i), uel=0.1,
             wp=float(i) - 0.1, lambda_c=4.0e-57)
        for i in range(1, 5)
    ]
    metrics = _metrics(rows)
    assert metrics is not None
    assert metrics["criteria_pass"]
    assert metrics["criteria"]["projected_cleavage_action_safe"]
    assert not metrics["legacy_negligible_positive_tip_J"]
    assert not metrics["legacy_negligible_tip_stress"]
    assert metrics["J_and_sigma_zero_gates_are_diagnostic_only"]
    assert metrics["projected_hazard_fraction_of_remaining_budget"] < 1.0e-50


def test_collapsed_stiffness_accessible_positive_hazard_fails() -> None:
    rows = [
        _row(i, force=4000.0, wext=float(i), uel=0.1,
             wp=float(i) - 0.1, lambda_c=1.0e-1)
        for i in range(1, 5)
    ]
    metrics = _metrics(rows)
    assert metrics is not None
    assert not metrics["criteria_pass"]
    assert not metrics["criteria"]["projected_cleavage_action_safe"]
    assert metrics["projected_hazard_fraction_of_remaining_budget"] > 0.01
