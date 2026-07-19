"""v10.2.14 physical signed-slip perturbation evaluator.

The ribbon endpoint is used exactly.  Triangle/ribbon overlap is integrated by
area fraction, so a graded mesh no longer requires a centroid on the slip ray
and distinct MPZ bins cannot silently collapse onto one FEM endpoint.  The
source remains at the crack surface, stiffness-killed overlap is removed, and a
nonzero mechanically supported terminal neighborhood is required.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .interaction_integral_v10214 import (
    MODEL_ID as INTERACTION_INTEGRAL_MODEL_ID,
    compute_signed_interaction_integral,
)
from .slip_ribbon_overlap_v10214 import (
    RibbonOverlapSupport,
    overlap_weighted_slip_ribbon_increment,
)
from .unit_slip_perturbation_v1026 import (
    MODEL_ID as RIBBON_MODEL_ID,
    SlipRibbonPerturbation,
    solve_fixed_crack_state,
)

MODEL_ID = "v10.2.14_physical_signed_slip_exact_overlap_intrinsic_isotropy"
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
    kappa = float(stiffness_kappa)
    if not np.isfinite(kappa) or kappa < 0.0:
        raise ValueError("stiffness_kappa must be finite and nonnegative")
    de = np.clip(_element_damage(mesh, d), 0.0, 1.0)
    return (1.0 - de) ** 2 + kappa


def _mask_killed_ribbon_elements(
    mesh,
    d: np.ndarray,
    increment: np.ndarray,
    support: RibbonOverlapSupport,
    *,
    minimum_residual_stiffness_fraction: float,
    stiffness_kappa: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    value = np.asarray(increment, dtype=float).copy()
    if value.ndim != 2 or value.shape[1] != int(mesh.ne):
        raise ValueError("slip-ribbon increment must have shape (nstrain, mesh.ne)")
    overlap = np.asarray(support.overlap_area_e_m2, dtype=float)
    terminal_overlap = np.asarray(support.terminal_overlap_area_e_m2, dtype=float)
    if overlap.shape != (int(mesh.ne),) or terminal_overlap.shape != (int(mesh.ne),):
        raise ValueError("ribbon-overlap support is incompatible with the FEM mesh")
    selected_before = overlap > 0.0
    if not np.any(selected_before):
        raise ValueError("signed slip ribbon has no geometric FEM overlap")

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
    supported = selected_before & ~killed
    if not np.any(supported):
        raise ValueError(
            "signed slip ribbon has no mechanically supported FEM overlap after "
            "stiffness masking"
        )
    supported_terminal_area = float(np.sum(terminal_overlap[~killed]))
    if supported_terminal_area <= 1.0e-30:
        raise ValueError(
            "signed slip ribbon endpoint has no mechanically supported FEM overlap; "
            "the requested MPZ station is not represented by the production mesh"
        )

    geometric_area = float(np.sum(overlap))
    supported_area = float(np.sum(overlap[~killed]))
    killed_area = float(np.sum(overlap[killed]))
    terminal_geometric_area = float(np.sum(terminal_overlap))
    return value, {
        "stiffness_mask_model": "g=(1-element_mean_damage)^2+kappa",
        "stiffness_kappa": float(stiffness_kappa),
        "minimum_residual_stiffness_fraction": threshold,
        "selected_elements_before_stiffness_mask": int(
            np.count_nonzero(selected_before)
        ),
        "selected_elements_after_stiffness_mask": int(np.count_nonzero(supported)),
        "stiffness_killed_selected_elements": int(
            np.count_nonzero(selected_before & killed)
        ),
        "geometric_overlap_area_before_stiffness_mask_m2": geometric_area,
        "supported_overlap_area_after_stiffness_mask_m2": supported_area,
        "stiffness_killed_overlap_area_m2": killed_area,
        "stiffness_killed_overlap_area_fraction": killed_area
        / max(geometric_area, 1.0e-30),
        "terminal_geometric_overlap_area_m2": terminal_geometric_area,
        "terminal_supported_overlap_area_m2": supported_terminal_area,
        "terminal_supported_fraction": supported_terminal_area
        / max(terminal_geometric_area, 1.0e-30),
        "minimum_supported_residual_stiffness_fraction": float(
            np.min(residual[supported])
        ),
        "maximum_supported_residual_stiffness_fraction": float(
            np.max(residual[supported])
        ),
        "surface_overlap_is_clipped_not_relocated": True,
        "partially_damaged_load_bearing_elements_retained": True,
        "mechanically_supported_terminal_required": True,
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
    raw_increment, perturbation_audit, support = (
        overlap_weighted_slip_ribbon_increment(mesh, perturbation)
    )
    increment, stiffness_audit = _mask_killed_ribbon_elements(
        mesh,
        d,
        raw_increment,
        support,
        minimum_residual_stiffness_fraction=(
            minimum_residual_stiffness_fraction
        ),
        stiffness_kappa=stiffness_kappa,
    )
    requested_area = max(
        float(perturbation_audit["requested_ribbon_area_m2"]), 1.0e-30
    )
    supported_area = float(
        stiffness_audit["supported_overlap_area_after_stiffness_mask_m2"]
    )
    perturbation_audit = {
        **perturbation_audit,
        **stiffness_audit,
        "mesh_area_ratio": supported_area / requested_area,
        "mesh_area_ratio_semantics": (
            "mechanically_supported_exact_overlap_area/requested_ribbon_area"
        ),
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
    common = dict(
        mesh=mesh,
        d=d,
        crack_tip=crack_tip,
        crack_direction=crack_direction,
        mat=mat,
        ell=interaction_ell_m,
        cfg=interaction_cfg,
        crack_segments=crack_segments,
        exclude_radius=exclude_radius_m,
        D=D,
    )
    base_K = compute_signed_interaction_integral(
        u=base_state["u"],
        sigma_gp=base_state["sigma_gp"],
        **common,
    )
    perturbed_K = compute_signed_interaction_integral(
        u=perturbed["u"],
        sigma_gp=perturbed["sigma_gp"],
        **common,
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
        "K_II_base_Pa_sqrt_m": float(base_K.K_II_base_Pa_sqrt_m)
        if hasattr(base_K, "K_II_base_Pa_sqrt_m")
        else float(base_K.K_II_Pa_sqrt_m),
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
