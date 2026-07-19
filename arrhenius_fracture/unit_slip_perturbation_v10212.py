"""v10.2.12 physical signed-slip perturbation evaluator.

This module reuses the mechanically normalized slip ribbon from v10.2.6 but
forces every base and perturbed SIF extraction through the hardened v10.2.9
analytic-gradient interaction integral.  It never modifies the accepted
production state and never uses a fitted population-to-shielding coefficient.

A ribbon may start at a stiffness-killed crack/free surface.  Elements that lie
outside the surviving body at that source end are clipped from the imposed
eigenstrain.  Clipping is permitted only in a short source-side zone; damaged
material in the ribbon interior or at the terminal dislocation remains a hard
error.
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

MODEL_ID = "v10.2.13_physical_signed_slip_source_surface_clipping"


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
    if damage.size == int(mesh.nn):
        return np.mean(damage[np.asarray(mesh.elems, dtype=int)], axis=1)
    if damage.size == int(mesh.ne):
        return damage.copy()
    raise ValueError("damage field size is incompatible with the FEM mesh")


def _clip_source_surface_overlap(
    mesh,
    d: np.ndarray,
    increment: np.ndarray,
    perturbation: SlipRibbonPerturbation,
    *,
    maximum_damaged_area_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove only source-side ribbon area that lies outside the intact body.

    Source-side clipping is applied before terminal validation.  This ordering is
    essential for the first FEM-resolved station, where the discrete source and
    terminal support windows can touch.  Damage in that overlap belongs to the
    crack/free-surface source and is clipped; damaged support strictly beyond the
    source allowance remains a hard terminal/interior error.
    """
    allowed_damage = float(maximum_damaged_area_fraction)
    if not 0.0 <= allowed_damage < 1.0:
        raise ValueError("maximum_damaged_area_fraction must lie in [0,1)")

    corrected = np.asarray(increment, dtype=float).copy()
    selected = np.any(np.abs(corrected) > 0.0, axis=0)
    if not np.any(selected):
        raise ValueError("signed slip ribbon selects no elements")

    element_damage = _element_damage(mesh, d)
    area = np.asarray(mesh.area_e, dtype=float)
    raw_area = float(np.sum(area[selected]))
    if raw_area <= 0.0:
        raise ValueError("signed slip ribbon has zero represented area")
    raw_damage_fraction = float(
        np.sum(area[selected] * np.clip(element_damage[selected], 0.0, 1.0))
        / raw_area
    )

    p = perturbation.validate()
    start = np.asarray(p.start_xy_m, dtype=float).reshape(2)
    end = np.asarray(p.end_xy_m, dtype=float).reshape(2)
    tangent = end - start
    length = float(np.linalg.norm(tangent))
    tangent /= length
    centroids = np.asarray(mesh.nodes, dtype=float)[
        np.asarray(mesh.elems, dtype=int)
    ].mean(axis=1)
    longitudinal = (centroids - start[None, :]) @ tangent

    damaged = selected & (element_damage > allowed_damage)
    h_tip = max(float(getattr(mesh, "hbar_tip", 0.0)), 0.0)
    width = float(p.width_m)
    source_limit = min(
        0.75 * length,
        max(2.0 * width, 3.0 * h_tip, 1.0e-12),
    )
    terminal_span = min(
        0.25 * length,
        max(0.5 * width, h_tip, 1.0e-12),
    )
    terminal = selected & (longitudinal >= length - terminal_span)
    if not np.any(terminal):
        raise ValueError("signed slip ribbon has no FEM support at its terminal line")

    # A damaged terminal element that is also inside the source allowance is a
    # discrete representation of the crack/free surface and is clipped below.
    # Only terminal damage strictly beyond that allowance represents a damaged
    # terminal dislocation.
    tolerance = max(1.0e-12, 1.0e-10 * max(length, width, h_tip, 1.0))
    terminal_damaged_outside_source = (
        damaged
        & terminal
        & (longitudinal > source_limit + tolerance)
    )
    if np.any(terminal_damaged_outside_source):
        raise ValueError(
            "signed slip ribbon terminal lies in stiffness-killed crack material"
        )

    interior_damaged = (
        damaged
        & (longitudinal > source_limit + tolerance)
        & ~terminal
    )
    if np.any(interior_damaged):
        worst = float(np.max(longitudinal[interior_damaged]))
        raise ValueError(
            "signed slip ribbon crosses stiffness-killed crack material away from "
            f"its source: furthest damaged longitudinal position={worst:.6g} m, "
            f"source allowance={source_limit:.6g} m"
        )

    source_clipped = damaged & (longitudinal <= source_limit + tolerance)
    corrected[:, source_clipped] = 0.0
    retained = np.any(np.abs(corrected) > 0.0, axis=0)
    if not np.any(retained):
        raise ValueError("source-surface clipping removed the complete slip ribbon")

    retained_terminal = retained & terminal
    if not np.any(retained_terminal):
        raise ValueError("source-surface clipping removed terminal ribbon support")
    if np.any(element_damage[retained_terminal] > allowed_damage):
        raise ValueError(
            "signed slip ribbon terminal lies in stiffness-killed crack material"
        )

    retained_area = float(np.sum(area[retained]))
    residual_damage_fraction = float(
        np.sum(area[retained] * np.clip(element_damage[retained], 0.0, 1.0))
        / retained_area
    )
    if residual_damage_fraction > allowed_damage + 1.0e-12:
        raise ValueError(
            "signed slip ribbon retains excessive damaged material after source "
            f"clipping: fraction={residual_damage_fraction:.6g}, "
            f"allowed={allowed_damage:.6g}"
        )

    clipped_area = float(np.sum(area[source_clipped]))
    return corrected, {
        "raw_selected_elements": int(np.sum(selected)),
        "selected_elements_after_source_clipping": int(np.sum(retained)),
        "raw_represented_area_m2": raw_area,
        "represented_area_after_source_clipping_m2": retained_area,
        "raw_damaged_area_fraction": raw_damage_fraction,
        "damaged_area_fraction": residual_damage_fraction,
        "maximum_allowed_damaged_area_fraction": allowed_damage,
        "source_surface_clipping_applied": bool(np.any(source_clipped)),
        "source_surface_clipped_elements": int(np.sum(source_clipped)),
        "source_surface_clipped_area_m2": clipped_area,
        "source_surface_clipped_area_fraction": clipped_area / raw_area,
        "source_surface_longitudinal_allowance_m": source_limit,
        "terminal_support_span_m": terminal_span,
        "terminal_in_intact_material": True,
        "interior_crossing_rejected": True,
        "ribbon_in_intact_material": True,
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
    maximum_damaged_area_fraction: float = 0.05,
) -> dict[str, Any]:
    """Evaluate one signed line perturbation about an equilibrated fixed crack."""
    increment, perturbation_audit = slip_ribbon_eigenstrain_increment(
        mesh, perturbation
    )
    increment, clipping_audit = _clip_source_surface_overlap(
        mesh,
        d,
        increment,
        perturbation,
        maximum_damaged_area_fraction=maximum_damaged_area_fraction,
    )
    represented_area = clipping_audit[
        "represented_area_after_source_clipping_m2"
    ]
    requested_area = max(
        float(perturbation_audit.get("requested_ribbon_area_m2", 0.0)),
        1.0e-30,
    )
    perturbation_audit = {
        **perturbation_audit,
        **clipping_audit,
        "selected_elements": clipping_audit[
            "selected_elements_after_source_clipping"
        ],
        "represented_area_m2": represented_area,
        "mesh_area_ratio": represented_area / requested_area,
        "normalization_after_surface_clipping": (
            "requested signed line content unchanged; killed crack-surface "
            "elements carry no eigenstrain"
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
