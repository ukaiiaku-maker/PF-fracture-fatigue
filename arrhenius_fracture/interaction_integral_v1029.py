"""Hardened signed mixed-mode interaction integral for v10.2.9.

This implementation keeps the v10.2.6 isotropic plane-strain convention but
removes two avoidable numerical artifacts from kernel generation:

* Williams auxiliary displacement gradients are evaluated analytically rather
  than by a radius-dependent finite difference;
* the annular domain weight uses a C1 cubic Hermite transition, so its radial
  derivative vanishes at both contour boundaries.

The routine remains fail-closed for materially anisotropic stiffness.  It is
intended for candidate-independent signed slip-perturbation calculations, not
as an anisotropic Stroh interaction integral.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .config import ElasticProperties, JIntegralConfig
from .interaction_integral_v1026 import (
    InteractionIntegralResult,
    _auxiliary_displacement_local,
    _auxiliary_stress_local,
    _auxiliary_strain_from_stress,
    isotropic_plane_strain_D,
    require_isotropic_stiffness,
)
from .j_integral import _line_of_sight_blocked

MODEL_ID = "v10.2.9_signed_isotropic_interaction_integral_analytic"


def _hermite_plateau_q(
    r: np.ndarray, r_inner: float, r_outer: float
) -> np.ndarray:
    """C1 annular weight: one inside, zero outside, flat at both joins."""
    radius = np.asarray(r, dtype=float)
    if not (0.0 < float(r_inner) < float(r_outer)):
        raise ValueError("interaction-integral radii require 0 < inner < outer")
    q = np.ones_like(radius)
    q[radius >= r_outer] = 0.0
    mask = (radius > r_inner) & (radius < r_outer)
    t = (radius[mask] - r_inner) / (r_outer - r_inner)
    q[mask] = 1.0 - 3.0 * t**2 + 2.0 * t**3
    return q


def _auxiliary_gradient_local_analytic(
    x: float,
    y: float,
    mode: str,
    E_Pa: float,
    poisson: float,
    K_aux_Pa_sqrt_m: float = 1.0,
) -> np.ndarray:
    """Return du_i/dx_j for the isotropic Williams auxiliary displacement.

    The displacement has the form ``u=A*sqrt(r)*F(theta)``.  Applying the polar
    chain rule gives

    ``du/dx=A/sqrt(r)*(0.5*F*cos(theta)-F'*sin(theta))`` and
    ``du/dy=A/sqrt(r)*(0.5*F*sin(theta)+F'*cos(theta))``.
    """
    r = float(np.hypot(x, y))
    if r <= 0.0:
        raise ValueError("auxiliary crack field is singular at the crack tip")
    theta = float(np.arctan2(y, x))
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    ch = float(np.cos(0.5 * theta))
    sh = float(np.sin(0.5 * theta))
    kappa = 3.0 - 4.0 * float(poisson)
    mu = float(E_Pa) / (2.0 * (1.0 + float(poisson)))
    A_over_sqrt_r = (
        float(K_aux_Pa_sqrt_m)
        / (2.0 * mu * np.sqrt(2.0 * np.pi * r))
    )

    if mode == "I":
        g = kappa - c
        F = np.array([ch * g, sh * g], dtype=float)
        Fp = np.array(
            [
                -0.5 * sh * g + ch * s,
                0.5 * ch * g + sh * s,
            ],
            dtype=float,
        )
    elif mode == "II":
        gx = kappa + 2.0 + c
        gy = kappa - 2.0 + c
        F = np.array([sh * gx, -ch * gy], dtype=float)
        Fp = np.array(
            [
                0.5 * ch * gx - sh * s,
                0.5 * sh * gy + ch * s,
            ],
            dtype=float,
        )
    else:
        raise ValueError(f"invalid auxiliary mode {mode!r}")

    du_dx = A_over_sqrt_r * (0.5 * F * c - Fp * s)
    du_dy = A_over_sqrt_r * (0.5 * F * s + Fp * c)
    return np.column_stack([du_dx, du_dy])


def _interaction_for_mode(
    *,
    mode: str,
    mesh,
    u: np.ndarray,
    sigma_gp: np.ndarray,
    d: np.ndarray,
    tip: np.ndarray,
    rotation: np.ndarray,
    q_node: np.ndarray,
    mat: ElasticProperties,
    crack_segments,
    r_inner: float,
    r_outer: float,
    exclude_radius: float,
) -> tuple[float, dict[str, Any]]:
    value = 0.0
    active = 0
    skipped_branch_cut = 0
    skipped_damage = 0
    skipped_los = 0
    near_segments = None
    if crack_segments:
        rmax2 = (3.0 * r_outer) ** 2
        near_segments = [
            (p0, p1)
            for p0, p1 in crack_segments
            if min(
                (p0[0] - tip[0]) ** 2 + (p0[1] - tip[1]) ** 2,
                (p1[0] - tip[0]) ** 2 + (p1[1] - tip[1]) ** 2,
            )
            < rmax2
        ]
        if not near_segments:
            near_segments = None

    nodal_displacement = np.asarray(u, dtype=float).reshape(-1, 2)
    for element in range(mesh.ne):
        conn = mesh.elems[element]
        qe = q_node[conn]
        if np.max(np.abs(qe)) < 1.0e-14 or np.ptp(qe) < 1.0e-14:
            continue
        if float(np.mean(d[conn])) > 0.95:
            skipped_damage += 1
            continue
        centroid = np.mean(mesh.nodes[conn], axis=0)
        local = rotation.T @ (centroid - tip)
        radius = float(np.hypot(local[0], local[1]))
        if radius <= max(float(exclude_radius), 1.0e-14):
            continue
        if local[0] < 0.0 and abs(local[1]) <= 1.0e-12 * max(radius, 1.0):
            skipped_branch_cut += 1
            continue
        if near_segments is not None and _line_of_sight_blocked(
            (float(tip[0]), float(tip[1])),
            (float(centroid[0]), float(centroid[1])),
            near_segments,
            r_inner,
        ):
            skipped_los += 1
            continue

        dq_global = mesh.dNdx_e[element] @ qe
        dq_local = rotation.T @ dq_global
        grad_global = nodal_displacement[conn].T @ mesh.dNdx_e[element].T
        grad_actual = rotation.T @ grad_global @ rotation
        sig = sigma_gp[:, element]
        sigma_global = np.array(
            [[sig[0], sig[2]], [sig[2], sig[1]]], dtype=float
        )
        sigma_actual = rotation.T @ sigma_global @ rotation

        sigma_aux = _auxiliary_stress_local(local[0], local[1], mode)
        grad_aux = _auxiliary_gradient_local_analytic(
            local[0], local[1], mode, mat.E, mat.nu
        )
        strain_aux = _auxiliary_strain_from_stress(
            sigma_aux, mat.E, mat.nu
        )
        interaction_energy = float(np.sum(sigma_actual * strain_aux))

        flux = np.zeros(2, dtype=float)
        for j in range(2):
            flux[j] = (
                float(np.dot(sigma_actual[:, j], grad_aux[:, 0]))
                + float(np.dot(sigma_aux[:, j], grad_actual[:, 0]))
                - (interaction_energy if j == 0 else 0.0)
            )
        value += float(np.dot(flux, dq_local)) * float(mesh.area_e[element])
        active += 1

    return value, {
        "mode": mode,
        "active_elements": active,
        "skipped_damage_elements": skipped_damage,
        "skipped_branch_cut_elements": skipped_branch_cut,
        "skipped_line_of_sight_elements": skipped_los,
        "r_inner_m": float(r_inner),
        "r_outer_m": float(r_outer),
    }


def compute_signed_interaction_integral(
    mesh,
    u: np.ndarray,
    sigma_gp: np.ndarray,
    d: np.ndarray,
    crack_tip: np.ndarray,
    crack_direction: np.ndarray,
    mat: ElasticProperties,
    ell: float,
    *,
    cfg: JIntegralConfig | None = None,
    crack_segments=None,
    exclude_radius: float = 0.0,
    D: np.ndarray | None = None,
    isotropy_relative_tolerance: float = 1.0e-8,
) -> InteractionIntegralResult:
    """Return signed KI and KII using analytic auxiliary gradients."""
    require_isotropic_stiffness(
        D, mat, relative_tolerance=isotropy_relative_tolerance
    )
    if cfg is None:
        cfg = JIntegralConfig()
    tip = np.asarray(crack_tip, dtype=float).reshape(2)
    direction = np.asarray(crack_direction, dtype=float).reshape(2)
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-14:
        raise ValueError("crack direction must be nonzero")
    e1 = direction / norm
    e2 = np.array([-e1[1], e1[0]], dtype=float)
    rotation = np.column_stack([e1, e2])
    r_inner = float(cfg.r_inner_factor) * float(ell)
    r_outer = float(cfg.r_outer_factor) * float(ell)
    radii = np.linalg.norm(np.asarray(mesh.nodes) - tip[None, :], axis=1)
    q_node = _hermite_plateau_q(radii, r_inner, r_outer)

    common = dict(
        mesh=mesh,
        u=u,
        sigma_gp=sigma_gp,
        d=d,
        tip=tip,
        rotation=rotation,
        q_node=q_node,
        mat=mat,
        crack_segments=crack_segments,
        r_inner=r_inner,
        r_outer=r_outer,
        exclude_radius=exclude_radius,
    )
    M_I, diag_I = _interaction_for_mode(mode="I", **common)
    M_II, diag_II = _interaction_for_mode(mode="II", **common)
    factor = float(mat.Eprime) / 2.0
    return InteractionIntegralResult(
        K_I_Pa_sqrt_m=float(factor * M_I),
        K_II_Pa_sqrt_m=float(factor * M_II),
        M_I_m_per_Pa=float(M_I),
        M_II_m_per_Pa=float(M_II),
        diagnostics={
            "schema": MODEL_ID,
            "plane_strain": True,
            "signed_modes": ["I", "II"],
            "auxiliary_K_Pa_sqrt_m": 1.0,
            "isotropic_auxiliary_fields": True,
            "auxiliary_displacement_gradient": "analytic_polar_chain_rule",
            "domain_weight": "cubic_Hermite_C1",
            "isotropy_relative_tolerance": float(isotropy_relative_tolerance),
            "mode_I": diag_I,
            "mode_II": diag_II,
        },
    )


__all__ = [
    "MODEL_ID",
    "InteractionIntegralResult",
    "compute_signed_interaction_integral",
    "isotropic_plane_strain_D",
    "require_isotropic_stiffness",
    "_hermite_plateau_q",
    "_auxiliary_gradient_local_analytic",
    "_auxiliary_displacement_local",
    "_auxiliary_stress_local",
]
