"""v10.2.14 physical signed-slip perturbation evaluator.

The fixed-crack FEM uses a nodal damage field only to degrade stiffness through
``g=(1-d_e)^2+kappa``.  A surface-emitted slip ribbon is therefore allowed to
start on the crack surface and to overlap the fully killed crack band, but the
represented eigenstrain is applied only in elements that still carry a
specified minimum residual stiffness.  This preserves the intended
surface-terminated dislocation topology without treating partially damaged,
load-bearing process-zone elements as voids.
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

MODEL_ID = "v10.2.14_physical_signed_slip_stiffness_masked_analytic_interaction_integral"
DEFAULT_STIFFNESS_KAPPA = 1.0e-6
DEFAULT_MINIMUM_RESIDUAL_STIFFNESS_FRACTION = 1.0e-3


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


def _element_damage(mesh, d: np.ndarray) -> np.ndarray:
    damage = np.asarray(d, dtype=float).reshape(-1)
    elems = np.asarray(mesh.elems, dtype=int)
    if damage.size == int(mesh.nn):
        return np.mean(damage[elems], axis=1)
    if damage.size == int(mesh.ne):
        return damage.copy()
    raise ValueError("damage field size is incompatible with the FEM mesh")


def element_residual_stiffness_fraction(
    mesh,
    d: np.ndarray,
    *,
    stiffness_kappa: float = DEFAULT_STIFFNESS_KAPPA,
) -> np.ndarray:
    """Return the exact scalar degradation used by ``assemble_mechanics``."""
    kappa = float(stiffness_kappa)
    if not np.isfinite(kappa) or kappa < 0.0:
        raise ValueError("stiffness_kappa must be finite and nonnegative")
    de = np.clip(_element_damage(mesh, d), 0.0, 1.0)
    return (1.0 - de) ** 2 + kappa


def _mask_killed_ribbon_elements(
    mesh,
    d: np.ndarray,
    increment: np.ndarray,
    *,
    minimum_residual_stiffness_fraction: float,
    stiffness_kappa: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove eigenstrain only from mechanically killed elements.

    The geometric ribbon may originate on a crack surface and may therefore
    overlap the killed crack band.  Those elements already have negligible
    stiffness and cannot represent physical line content.  Partially damaged
    elements are retained according to the same degradation law as the FEM.
    """
    value = np.asarray(increment, dtype=float).copy()
    if value.ndim != 2 or value.shape[1] != int(mesh.ne):
        raise ValueError("slip-ribbon increment must have shape (nstrain, mesh.ne)")
    selected_before = np.any(np.abs(value) > 0.0, axis=0)
    if not np.any(selected_before):
        raise ValueError("signed slip ribbon selects no elements")

    threshold = float(minimum_residual_stiffness_fraction)
    if not np.isfinite(threshold) or not 0.0 <= threshold < 1.0:
        raise ValueError(
            "minimum_residual_stiffness_fraction must lie in [0,1)"
        )
    residual = element_residual_stiffness_fraction(
        mesh, d, stiffness_kappa=stiffness_kappa
    )
    killed = residual < threshold
    value[:, killed] = 0.0
    selected_after = np.any(np.abs(value) > 0.0, axis=0)
    if not np.any(selected_after):
        raise ValueError(
            "signed slip ribbon has no mechanically supported FEM elements after "
            "stiffness masking"
        )

    area = np.asarray(mesh.area_e, dtype=float)
    area_before = float(np.sum(area[selected_before]))
    area_after = float(np.sum(area[selected_after]))
    killed_area = float(np.sum(area[selected_before & killed]))
    supported_residual = residual[selected_after]
    return value, {
        "stiffness_mask_model": "g=(1-element_mean_damage)^2+kappa",
        "stiffness_kappa": float(stiffness_kappa),
        "minimum_residual_stiffness_fraction": threshold,
        "selected_elements_before_stiffness_mask": int(
            np.count_nonzero(selected_before)
        ),
        "selected_elements_after_stiffness_mask": int(
            np.count_nonzero(selected_after)
        ),
        "stiffness_killed_selected_elements": int(
            np.count_nonzero(selected_before & killed)
        ),
        "selected_area_before_stiffness_mask_m2": area_before,
        "selected_area_after_stiffness_mask_m2": area_after,
        "stiffness_killed_selected_area_m2": killed_area,
        "stiffness_killed_selected_area_fraction": (
            killed_area / area_before if area_before > 0.0 else 0.0
        ),
        "minimum_supported_residual_stiffness_fraction": float(
            np.min(supported_residual)
        ),
        "maximum_supported_residual_stiffness_fraction": float(
            np.max(supported_residual)
        ),
        "surface_overlap_is_clipped_not_relocated": True,
        "partially_damaged_load_bearing_elements_retained": True,
    }


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
    minimum_residual_stiffness_fraction: float = (
        DEFAULT_MINIMUM_RESIDUAL_STIFFNESS_FRACTION
    ),
    stiffness_kappa: float = DEFAULT_STIFFNESS_KAPPA,
) -> dict[str, Any]:
    """Evaluate one signed line perturbation about an equilibrated fixed crack."""
    raw_increment, perturbation_audit = slip_ribbon_eigenstrain_increment(
        mesh, perturbation
    )
    increment, stiffness_audit = _mask_killed_ribbon_elements(
        mesh,
        d,
        raw_increment,
        minimum_residual_stiffness_fraction=(
            minimum_residual_stiffness_fraction
        ),
        stiffness_kappa=stiffness_kappa,
    )
    perturbation_audit = {**perturbation_audit, **stiffness_audit}

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
    "DEFAULT_STIFFNESS_KAPPA",
    "DEFAULT_MINIMUM_RESIDUAL_STIFFNESS_FRACTION",
    "element_residual_stiffness_fraction",
    "equilibrated_base_state",
    "interaction_response",
]
