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
) -> dict[str, Any]:
    """Evaluate one signed line perturbation about an equilibrated fixed crack."""
    increment, perturbation_audit = slip_ribbon_eigenstrain_increment(
        mesh, perturbation
    )
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
