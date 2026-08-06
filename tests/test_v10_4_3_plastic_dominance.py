from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pytest

from arrhenius_fracture.plastic_dominance_v1043 import (
    contour_scan_v1043,
    terminal_metrics_v1043,
    transform_source,
)


ROOT = Path(__file__).resolve().parents[1]


class Args:
    plastic_flow_window_steps = 4
    plastic_flow_min_step = 4
    plastic_flow_max_da_fraction = 0.1
    plastic_flow_min_plastic_fraction = 0.50
    plastic_flow_min_cumulative_plastic_fraction = 0.10
    plastic_flow_max_elastic_fraction = 0.50
    plastic_flow_max_tangent_fraction = 0.50
    plastic_flow_energy_balance_tolerance = 0.01


def _window(
    *,
    phi: float,
    elastic: float,
    active: float,
    stagger: float,
    plastic_work_fraction: float,
    energy_multiplier: float = 1.0,
    n_fire: int = 0,
) -> deque:
    rows = deque(maxlen=4)
    for step in range(1, 5):
        Wext = float(step)
        Wp = plastic_work_fraction * Wext * energy_multiplier
        Uel = (1.0 - plastic_work_fraction) * Wext
        rows.append(
            {
                "step": step,
                "Uapp": float(step),
                "Ftop": 1.0 - 0.2 * step,
                "J_positive": 2500.0,
                "J_signed": 2500.0,
                "sigma_tip": 1.0e9,
                "B": 1.0e-5 * step,
                "lambda_c": 1.0e-8,
                "n_fire": n_fire,
                "a_tip": 5.0e-4,
                "W_ext": Wext,
                "U_el": Uel,
                "W_p": Wp,
                "W_emit": 0.0,
                "plastic_accommodation_ratio": phi,
                "elastic_accommodation_ratio": elastic,
                "active_plastic_area_fraction": active,
                "stagger_relative_change": stagger,
            }
        )
    return rows


def _metrics(window, *, cumulative_Wp: float, cumulative_Uel: float):
    return terminal_metrics_v1043(
        window,
        Args(),
        Eprime=4.0e11,
        da_phys=5.0e-6,
        sigma_reference=3.0e10,
        peak_J_positive=1.0e5,
        peak_force=1.0,
        stiffness_reference=1.0,
        remaining_steps=1000,
        nominal_dt_s=8.4,
        cumulative_Wp=cumulative_Wp,
        cumulative_Uel=cumulative_Uel,
        cumulative_Wemit=0.0,
    )


def test_transformed_source_compiles_and_rebases_constitutive_time():
    source = (ROOT / "arrhenius_fracture" / "sharp_front.py").read_text()
    transformed = transform_source(source)
    compile(transformed, "sharp_front.py[v10.4.3-test]", "exec")
    assert "ep_gp_step0_v1043.copy()" in transformed
    assert "rho_gp_step0_v1043.copy()" in transformed
    assert "constitutive_dWp_accepted_gp_final_stagger_iterate" in transformed
    assert "plastic_flow_candidate_latest.json" in transformed
    assert "plastic_flow_candidate_history.jsonl" in transformed
    assert "v10.4.3_plastic_dominance_terminal_audit_v1" in transformed
    assert "plastic_terminal_is_model_limit_censor" in transformed
    assert "future_fracture_beyond_terminal_resolved" in transformed
    assert "J_eff = max(J_signed, 0.0)" in transformed


def test_sustained_majority_plastic_accommodation_passes():
    window = _window(
        phi=0.80,
        elastic=0.20,
        active=0.50,
        stagger=0.01,
        plastic_work_fraction=0.80,
    )
    result = _metrics(window, cumulative_Wp=3.2, cumulative_Uel=0.8)
    assert result is not None
    assert result["criteria_pass"] is True
    assert result["failed_criteria"] == []
    assert result["plastic_terminal_is_model_limit_censor"] is True
    assert result["future_fracture_beyond_terminal_resolved"] is False
    # Positive J is reported, not used as an arbitrary athermal terminal veto.
    assert result["J_tip_positive_max_window_J_per_m2"] == 2500.0


def test_nearly_elastic_load_bearing_state_fails():
    window = _window(
        phi=0.01,
        elastic=0.99,
        active=0.001,
        stagger=0.01,
        plastic_work_fraction=0.01,
    )
    result = _metrics(window, cumulative_Wp=0.04, cumulative_Uel=3.96)
    assert result is not None
    assert result["criteria_pass"] is False
    assert result["criteria"]["plastic_accommodation_dominant"] is False
    assert result["criteria"]["elastic_or_tangent_response_subdominant"] is False
    assert result["criteria"]["spatially_resolved_plastic_activity"] is False


def test_energy_imbalance_blocks_otherwise_plastic_terminal():
    window = _window(
        phi=0.80,
        elastic=0.20,
        active=0.50,
        stagger=0.01,
        plastic_work_fraction=0.80,
        energy_multiplier=1.25,
    )
    result = _metrics(window, cumulative_Wp=4.0, cumulative_Uel=0.8)
    assert result is not None
    assert result["criteria_pass"] is False
    assert result["criteria"]["energy_balance_bounded"] is False


def test_sharp_fracture_first_passage_wins_same_window():
    window = _window(
        phi=0.80,
        elastic=0.20,
        active=0.50,
        stagger=0.01,
        plastic_work_fraction=0.80,
        n_fire=1,
    )
    result = _metrics(window, cumulative_Wp=3.2, cumulative_Uel=0.8)
    assert result is not None
    assert result["criteria_pass"] is False
    assert result["criteria"]["no_sharp_fracture_first_passage"] is False


def test_stagger_nonconvergence_blocks_terminal():
    window = _window(
        phi=0.80,
        elastic=0.20,
        active=0.50,
        stagger=0.20,
        plastic_work_fraction=0.80,
    )
    result = _metrics(window, cumulative_Wp=3.2, cumulative_Uel=0.8)
    assert result is not None
    assert result["criteria_pass"] is False
    assert result["criteria"]["stagger_iteration_converged"] is False


def test_contour_scan_uses_positive_raw_signed_J_only():
    values = iter([4.0, -3.0])

    def fake_compute(
        mesh,
        u,
        sigma,
        psi,
        damage,
        tip,
        direction,
        mat,
        ell,
        crack_segments=None,
        exclude_radius=0.0,
    ):
        J = next(values)
        return abs(J), np.sqrt(abs(J) * 4.0), {
            "J_signed": J,
            "r_inner": ell,
            "r_outer": 8.0 * ell,
            "n_active_elements": 20,
        }

    records = contour_scan_v1043(
        fake_compute,
        mesh=None,
        u=None,
        sigma_gp=None,
        psi_gp=None,
        damage=None,
        tip_xy=[0.0, 0.0],
        direction=[1.0, 0.0],
        mat=None,
        base_ell_m=1.0,
        multipliers="1 2",
        crack_segments=None,
        exclude_radius_m=0.0,
        sign_reference=1.0,
    )
    assert records[0]["J_positive_root_convention_J_per_m2"] == 4.0
    assert records[1]["J_positive_root_convention_J_per_m2"] == 0.0


def test_contour_scan_rejects_legacy_negative_sign_reference():
    with pytest.raises(RuntimeError, match="J_sign_ref=1"):
        contour_scan_v1043(
            lambda *args, **kwargs: (0.0, 0.0, {"J_signed": 0.0}),
            mesh=None,
            u=None,
            sigma_gp=None,
            psi_gp=None,
            damage=None,
            tip_xy=[0.0, 0.0],
            direction=[1.0, 0.0],
            mat=None,
            base_ell_m=1.0,
            multipliers="1",
            crack_segments=None,
            exclude_radius_m=0.0,
            sign_reference=-1.0,
        )
