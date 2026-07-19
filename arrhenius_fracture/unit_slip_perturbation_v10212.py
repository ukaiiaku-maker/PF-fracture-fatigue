"""v10.2.12 physical signed-slip perturbation evaluator.

This module reuses the mechanically normalized slip ribbon from v10.2.6 but
forces every base and perturbed SIF extraction through the hardened v10.2.9
analytic-gradient interaction integral.  It never modifies the accepted
production state and never uses a fitted population-to-shielding coefficient.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .interaction_integral_v1029 import (
    MODEL_ID as INTERACTION_INTEGRAL_MODEL_ID,
    compute_signed_interaction_integral,
)
from .unit_slip_perturbation_v1026 import (
    MODEL_ID as RIBBON_MODEL_ID,
    SlipRibbonPerturbation,
    slip_ribbon_eigenstrain_increment,
    solve_fixed_crack_state,
)

MODEL_ID = "v10.2.12_physical_signed_slip_analytic_interaction_integral"


def equilibrated_base_state(
    *,
    mesh,
    boundary,
    baseline_u: np.ndarray,
    baseline_ep_gp: np.ndarray,
    rho_gp: np.ndarray,
    d: np.ndarray,
    D: np.ndarray,
    mat,
    Uy_top: float,
    Uy_bot: float,
    cohesive_network=None,
) -> dict[str, Any]:
    return solve_fixed_crack_state(
        mesh=mesh,
        boundary=boundary,
        u=baseline_u,
        ep_gp=baseline_ep_gp,
        rho_gp=rho_gp,
        d=d,
        D=D,
        mat=mat,
        Uy_top=Uy_top,
        Uy_bot=Uy_bot,
        cohesive_network=cohesive_network,
    )


def _ribbon_damage_fraction(mesh, d: np.ndarray, increment: np.ndarray) -> float:
    selected = np.any(np.abs(np.asarray(increment, dtype=float)) > 0.0, axis=0)
    if not np.any(selected):
        raise ValueError("signed slip ribbon selects no elements")
    damage = np.asarray(d, dtype=float).reshape(-1)
    if damage.size == int(mesh.nn):
        element_damage = np.mean(damage[np.asarray(mesh.elems, dtype=int)], axis=1)
    elif damage.size == int(mesh.ne):
        element_damage = damage
    else:
        raise ValueError("damage field size is incompatible with the FEM mesh")
    area = np.asarray(mesh.area_e, dtype=float)
    selected_area = float(np.sum(area[selected]))
    if selected_area <= 0.0:
        raise ValueError("signed slip ribbon has zero represented area")
    return float(np.sum(area[selected] * np.clip(element_damage[selected], 0.0, 1.0)) / selected_area)


def interaction_response(
    *,
    mesh,
    base_state: dict[str, Any],
    baseline_ep_gp: np.ndarray,
    rho_gp: np.ndarray,
    d: np.ndarray,
    D: np.ndarray,
    mat,
    boundary,
    Uy_top: float,
    Uy_bot: float,
    crack_tip: np.ndarray,
    crack_direction: np.ndarray,
    interaction_ell_m: float,
    perturbation: SlipRibbonPerturbation,
    interaction_cfg=None,
    crack_segments=None,
    exclude_radius_m: float = 0.0,
    cohesive_network=None,
    maximum_damaged_area_fraction: float = 0.05,
) -> dict[str, Any]:
    """Evaluate one signed line perturbation about an equilibrated fixed crack."""
    increment, perturbation_audit = slip_ribbon_eigenstrain_increment(
        mesh, perturbation
    )
    damaged_fraction = _ribbon_damage_fraction(mesh, d, increment)
    allowed_damage = float(maximum_damaged_area_fraction)
    if not 0.0 <= allowed_damage < 1.0:
        raise ValueError("maximum_damaged_area_fraction must lie in [0,1)")
    if damaged_fraction > allowed_damage:
        raise ValueError(
            "signed slip ribbon lies in stiffness-killed crack material: "
            f"damaged area fraction={damaged_fraction:.6g}, allowed={allowed_damage:.6g}"
        )
    perturbation_audit = {
        **perturbation_audit,
        "damaged_area_fraction": damaged_fraction,
        "maximum_allowed_damaged_area_fraction": allowed_damage,
        "ribbon_in_intact_material": True,
    }
    perturbed_ep = np.asarray(baseline_ep_gp, dtype=float) + increment
    perturbed = solve_fixed_crack_state(
        mesh=mesh,
        boundary=boundary,
        u=np.asarray(base_state["u"], dtype=float),
        ep_gp=perturbed_ep,
        rho_gp=rho_gp,
        d=d,
        D=D,
        mat=mat,
        Uy_top=Uy_top,
        Uy_bot=Uy_bot,
        cohesive_network=cohesive_network,
    )
    base_K = compute_signed_interaction_integral(
        mesh,
        base_state["u"],
        base_state["sigma_gp"],
        d,
        crack_tip,
        crack_direction,
        mat,
        interaction_ell_m,
        cfg=interaction_cfg,
        crack_segments=crack_segments,
        exclude_radius=exclude_radius_m,
        D=D,
    )
    perturbed_K = compute_signed_interaction_integral(
        mesh,
        perturbed["u"],
        perturbed["sigma_gp"],
        d,
        crack_tip,
        crack_direction,
        mat,
        interaction_ell_m,
        cfg=interaction_cfg,
        crack_segments=crack_segments,
        exclude_radius=exclude_radius_m,
        D=D,
    )
    content = float(perturbation.signed_line_content)
    if content == 0.0:
        raise ValueError("signed line content must be nonzero")
    return {
        "schema": MODEL_ID,
        "interaction_integral_schema": INTERACTION_INTEGRAL_MODEL_ID,
        "slip_ribbon_schema": RIBBON_MODEL_ID,
        "region": str(perturbation.region),
        "system": int(perturbation.system),
        "bin": int(perturbation.bin_index),
        "burgers_sign": 1 if content > 0.0 else -1,
        "delta_signed_line_content": content,
        "K_I_base_Pa_sqrt_m": float(base_K.K_I_Pa_sqrt_m),
        "K_I_perturbed_Pa_sqrt_m": float(perturbed_K.K_I_Pa_sqrt_m),
        "K_II_base_Pa_sqrt_m": float(base_K.K_II_Pa_sqrt_m),
        "K_II_perturbed_Pa_sqrt_m": float(perturbed_K.K_II_Pa_sqrt_m),
        "H_I_Pa_sqrt_m_per_signed_line": float(
            (base_K.K_I_Pa_sqrt_m - perturbed_K.K_I_Pa_sqrt_m) / content
        ),
        "H_II_Pa_sqrt_m_per_signed_line": float(
            (base_K.K_II_Pa_sqrt_m - perturbed_K.K_II_Pa_sqrt_m) / content
        ),
        "base_reaction_top": float(base_state["reaction_top"]),
        "perturbed_reaction_top": float(perturbed["reaction_top"]),
        "fixed_crack_geometry": True,
        "fixed_external_displacement": True,
        "production_state_not_mutated": True,
        "perturbation": perturbation_audit,
        "base_interaction_integral": base_K.diagnostics,
        "perturbed_interaction_integral": perturbed_K.diagnostics,
    }


__all__ = [
    "MODEL_ID",
    "INTERACTION_INTEGRAL_MODEL_ID",
    "SlipRibbonPerturbation",
    "equilibrated_base_state",
    "interaction_response",
]
