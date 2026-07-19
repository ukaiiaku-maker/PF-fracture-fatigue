"""Mechanically normalized signed slip perturbations for kernel generation.

A finite-width slip ribbon begins on the crack/free-surface side and terminates at
the target active/wake bin.  The terminal incompatibility represents the signed
edge-dislocation line sampled by the reduced model; the opposite termination is
absorbed at the crack/free surface.  For ribbon width ``w`` and plastic shear
``gamma``, the imposed displacement discontinuity is ``gamma*w`` and the signed
line content is

    delta_N_line = gamma * w / b.

Thus a requested signed line-content magnitude determines the eigenstrain
amplitude directly.  No toughness fit or arbitrary attenuation enters this
normalization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .interaction_integral_v1026 import compute_signed_interaction_integral

MODEL_ID = "v10.2.6_mechanically_normalized_signed_slip_ribbon"


@dataclass(frozen=True)
class SlipRibbonPerturbation:
    system: int
    region: str
    bin_index: int
    start_xy_m: np.ndarray
    end_xy_m: np.ndarray
    slip_direction: np.ndarray
    plane_normal: np.ndarray
    width_m: float
    burgers_m: float
    signed_line_content: float

    def validate(self) -> "SlipRibbonPerturbation":
        region = str(self.region).strip().lower()
        if region not in {"active", "wake"}:
            raise ValueError("slip-ribbon region must be active or wake")
        start = np.asarray(self.start_xy_m, dtype=float).reshape(2)
        end = np.asarray(self.end_xy_m, dtype=float).reshape(2)
        slip = np.asarray(self.slip_direction, dtype=float).reshape(2)
        normal = np.asarray(self.plane_normal, dtype=float).reshape(2)
        if np.any(~np.isfinite(start)) or np.any(~np.isfinite(end)):
            raise ValueError("slip-ribbon endpoints must be finite")
        length = float(np.linalg.norm(end - start))
        if length <= 0.0:
            raise ValueError("slip-ribbon length must be positive")
        if self.width_m <= 0.0 or self.burgers_m <= 0.0:
            raise ValueError("slip-ribbon width and Burgers magnitude must be positive")
        if float(np.linalg.norm(slip)) <= 0.0 or float(np.linalg.norm(normal)) <= 0.0:
            raise ValueError("slip direction and plane normal must be nonzero")
        slip = slip / np.linalg.norm(slip)
        normal = normal / np.linalg.norm(normal)
        if abs(float(np.dot(slip, normal))) > 1.0e-8:
            raise ValueError("slip direction and plane normal must be orthogonal")
        if not np.isfinite(self.signed_line_content) or self.signed_line_content == 0.0:
            raise ValueError("signed line content must be finite and nonzero")
        return self

    @property
    def plastic_shear(self) -> float:
        return (
            float(self.signed_line_content)
            * float(self.burgers_m)
            / float(self.width_m)
        )

    def audit_payload(self) -> dict[str, Any]:
        return {
            "schema": MODEL_ID,
            "system": int(self.system),
            "region": str(self.region),
            "bin": int(self.bin_index),
            "start_xy_m": np.asarray(self.start_xy_m, dtype=float).tolist(),
            "end_xy_m": np.asarray(self.end_xy_m, dtype=float).tolist(),
            "slip_direction": np.asarray(self.slip_direction, dtype=float).tolist(),
            "plane_normal": np.asarray(self.plane_normal, dtype=float).tolist(),
            "width_m": float(self.width_m),
            "burgers_m": float(self.burgers_m),
            "signed_line_content": float(self.signed_line_content),
            "plastic_shear": float(self.plastic_shear),
            "normalization": "delta_N_line = gamma * width / b",
            "fitted_to_toughness_or_fatigue": False,
        }


def slip_ribbon_eigenstrain_increment(
    mesh,
    perturbation: SlipRibbonPerturbation,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a Voigt engineering-strain increment on selected FEM elements."""
    p = perturbation.validate()
    start = np.asarray(p.start_xy_m, dtype=float).reshape(2)
    end = np.asarray(p.end_xy_m, dtype=float).reshape(2)
    tangent = end - start
    length = float(np.linalg.norm(tangent))
    tangent /= length
    centroids = np.asarray(mesh.nodes)[np.asarray(mesh.elems)].mean(axis=1)
    relative = centroids - start[None, :]
    longitudinal = relative @ tangent
    closest = start[None, :] + longitudinal[:, None] * tangent[None, :]
    transverse = np.linalg.norm(centroids - closest, axis=1)
    coordinate_scale = max(
        float(np.max(np.abs(start))),
        float(np.max(np.abs(end))),
        length,
        float(p.width_m),
        float(getattr(mesh, "hbar_tip", 0.0)),
        1.0,
    )
    selection_tolerance = max(
        128.0 * np.finfo(float).eps * coordinate_scale,
        1.0e-15,
    )
    selected = (
        (longitudinal >= -selection_tolerance)
        & (longitudinal <= length + selection_tolerance)
        & (transverse <= 0.5 * float(p.width_m) + selection_tolerance)
    )
    if not np.any(selected):
        raise ValueError(
            "slip ribbon selects no FEM elements; increase ribbon width or refine the mesh"
        )
    slip = np.asarray(p.slip_direction, dtype=float)
    slip /= np.linalg.norm(slip)
    normal = np.asarray(p.plane_normal, dtype=float)
    normal /= np.linalg.norm(normal)
    gamma = float(p.plastic_shear)
    # Symmetric plastic strain from beta^p = gamma s (x) n.  The third Voigt
    # component is engineering shear gamma_xy = 2 eps_xy.
    voigt = gamma * np.array(
        [
            slip[0] * normal[0],
            slip[1] * normal[1],
            slip[0] * normal[1] + slip[1] * normal[0],
        ],
        dtype=float,
    )
    increment = np.zeros((3, mesh.ne), dtype=float)
    increment[:, selected] = voigt[:, None]
    represented_area = float(np.sum(np.asarray(mesh.area_e)[selected]))
    return increment, {
        **p.audit_payload(),
        "selected_elements": int(np.sum(selected)),
        "represented_area_m2": represented_area,
        "represented_ribbon_length_m": length,
        "requested_ribbon_area_m2": length * float(p.width_m),
        "mesh_area_ratio": represented_area
        / max(length * float(p.width_m), 1.0e-30),
        "geometric_selection_tolerance_m": selection_tolerance,
        "endpoint_inclusion_is_tolerance_aware": True,
    }


def solve_fixed_crack_state(
    *,
    mesh,
    boundary,
    u: np.ndarray,
    ep_gp: np.ndarray,
    rho_gp: np.ndarray,
    d: np.ndarray,
    D: np.ndarray,
    mat,
    Uy_top: float,
    Uy_bot: float,
    cohesive_network=None,
) -> dict[str, Any]:
    """Equilibrate one fixed-crack state without evolving damage or plasticity."""
    from .fem import assemble_mechanics, solve_dirichlet, stress_state

    K, residual, _, _, _, _ = assemble_mechanics(
        mesh,
        np.asarray(u, dtype=float),
        np.asarray(ep_gp, dtype=float),
        np.asarray(rho_gp, dtype=float),
        np.asarray(d, dtype=float),
        np.asarray(D, dtype=float),
        mat,
        cohesive_network=cohesive_network,
    )
    u_equilibrium, reaction = solve_dirichlet(
        K,
        residual,
        np.asarray(u, dtype=float),
        boundary,
        float(Uy_top),
        float(Uy_bot),
    )
    sigma, sigma_eq, sigma1, psi = stress_state(
        mesh,
        u_equilibrium,
        np.asarray(ep_gp, dtype=float),
        np.asarray(d, dtype=float),
        np.asarray(D, dtype=float),
        mat,
    )
    return {
        "u": u_equilibrium,
        "reaction_top": float(reaction),
        "sigma_gp": sigma,
        "sigma_eq_gp": sigma_eq,
        "sigma1_gp": sigma1,
        "psi_gp": psi,
    }


def evaluate_signed_slip_perturbation(
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
    crack_tip: np.ndarray,
    crack_direction: np.ndarray,
    interaction_ell_m: float,
    perturbation: SlipRibbonPerturbation,
    interaction_cfg=None,
    crack_segments=None,
    exclude_radius_m: float = 0.0,
    cohesive_network=None,
) -> dict[str, Any]:
    """Evaluate baseline and one signed perturbation at fixed crack geometry."""
    base = solve_fixed_crack_state(
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
    increment, perturbation_audit = slip_ribbon_eigenstrain_increment(
        mesh, perturbation
    )
    perturbed_ep = np.asarray(baseline_ep_gp, dtype=float) + increment
    perturbed = solve_fixed_crack_state(
        mesh=mesh,
        boundary=boundary,
        u=base["u"],
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
        base["u"],
        base["sigma_gp"],
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
        "base_reaction_top": float(base["reaction_top"]),
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
    "SlipRibbonPerturbation",
    "slip_ribbon_eigenstrain_increment",
    "solve_fixed_crack_state",
    "evaluate_signed_slip_perturbation",
]
