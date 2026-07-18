"""Signed mixed-mode interaction integral for fixed-crack kernel generation.

This module evaluates signed ``K_I`` and ``K_II`` from a 2-D displacement/stress
state using isotropic Williams auxiliary fields.  It is intended for the
candidate-independent unit-slip/dislocation perturbation calculations used to
build the v10.2.6 shielding-kernel family.

The implementation is valid for isotropic plane strain.  The production W cubic
constants satisfy the isotropy condition when the Zener ratio is one.  A caller
that supplies an anisotropic stiffness matrix must first demonstrate that it is
numerically equivalent to the isotropic plane-strain matrix; otherwise this
routine fails closed rather than silently applying an invalid auxiliary field.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import ElasticProperties, JIntegralConfig
from .j_integral import _line_of_sight_blocked

MODEL_ID = "v10.2.6_signed_isotropic_interaction_integral"


@dataclass(frozen=True)
class InteractionIntegralResult:
    K_I_Pa_sqrt_m: float
    K_II_Pa_sqrt_m: float
    M_I_m_per_Pa: float
    M_II_m_per_Pa: float
    diagnostics: dict[str, Any]


def _plateau_q(r: np.ndarray, r_inner: float, r_outer: float) -> np.ndarray:
    if not (0.0 < r_inner < r_outer):
        raise ValueError("interaction-integral radii require 0 < inner < outer")
    q = np.ones_like(r, dtype=float)
    q[r >= r_outer] = 0.0
    mask = (r > r_inner) & (r < r_outer)
    q[mask] = (r_outer - r[mask]) / (r_outer - r_inner)
    return q


def isotropic_plane_strain_D(E_Pa: float, poisson: float) -> np.ndarray:
    E = float(E_Pa)
    nu = float(poisson)
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return c * np.array(
        [
            [1.0 - nu, nu, 0.0],
            [nu, 1.0 - nu, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - 2.0 * nu)],
        ],
        dtype=float,
    )


def require_isotropic_stiffness(
    D: np.ndarray | None,
    mat: ElasticProperties,
    *,
    relative_tolerance: float = 1.0e-8,
) -> None:
    if D is None:
        return
    supplied = np.asarray(D, dtype=float)
    reference = isotropic_plane_strain_D(mat.E, mat.nu)
    if supplied.shape != (3, 3) or not np.all(np.isfinite(supplied)):
        raise ValueError("interaction-integral stiffness must be a finite 3x3 matrix")
    scale = max(float(np.max(np.abs(reference))), 1.0)
    error = float(np.max(np.abs(supplied - reference))) / scale
    if error > float(relative_tolerance):
        raise ValueError(
            "v10.2.6 Williams interaction integral is isotropic; supplied stiffness "
            f"differs from isotropic plane strain by {error:.3e}"
        )


def _auxiliary_displacement_local(
    x: float,
    y: float,
    mode: str,
    E_Pa: float,
    poisson: float,
    K_aux_Pa_sqrt_m: float = 1.0,
) -> np.ndarray:
    r = float(np.hypot(x, y))
    if r <= 0.0:
        raise ValueError("auxiliary crack field is singular at the crack tip")
    theta = float(np.arctan2(y, x))
    mu = float(E_Pa) / (2.0 * (1.0 + float(poisson)))
    kappa = 3.0 - 4.0 * float(poisson)  # plane strain
    fac = (
        float(K_aux_Pa_sqrt_m)
        / (2.0 * mu)
        * np.sqrt(r / (2.0 * np.pi))
    )
    if mode == "I":
        bracket = kappa - np.cos(theta)
        return np.array(
            [
                fac * np.cos(0.5 * theta) * bracket,
                fac * np.sin(0.5 * theta) * bracket,
            ]
        )
    if mode == "II":
        return np.array(
            [
                fac
                * np.sin(0.5 * theta)
                * (kappa + 2.0 + np.cos(theta)),
                -fac
                * np.cos(0.5 * theta)
                * (kappa - 2.0 + np.cos(theta)),
            ]
        )
    raise ValueError(f"invalid auxiliary mode {mode!r}")


def _auxiliary_stress_local(
    x: float,
    y: float,
    mode: str,
    K_aux_Pa_sqrt_m: float = 1.0,
) -> np.ndarray:
    r = float(np.hypot(x, y))
    if r <= 0.0:
        raise ValueError("auxiliary crack field is singular at the crack tip")
    theta = float(np.arctan2(y, x))
    f = float(K_aux_Pa_sqrt_m) / np.sqrt(2.0 * np.pi * r)
    c = np.cos(0.5 * theta)
    s = np.sin(0.5 * theta)
    c3 = np.cos(1.5 * theta)
    s3 = np.sin(1.5 * theta)
    if mode == "I":
        sxx = f * c * (1.0 - s * s3)
        syy = f * c * (1.0 + s * s3)
        sxy = f * c * s * c3
    elif mode == "II":
        sxx = -f * s * (2.0 + c * c3)
        syy = f * s * c * c3
        sxy = f * c * (1.0 - s * s3)
    else:
        raise ValueError(f"invalid auxiliary mode {mode!r}")
    return np.array([[sxx, sxy], [sxy, syy]], dtype=float)


def _auxiliary_gradient_local(
    x: float,
    y: float,
    mode: str,
    E_Pa: float,
    poisson: float,
) -> np.ndarray:
    r = float(np.hypot(x, y))
    h = max(1.0e-5 * r, 1.0e-14)
    ux_plus = _auxiliary_displacement_local(x + h, y, mode, E_Pa, poisson)
    ux_minus = _auxiliary_displacement_local(x - h, y, mode, E_Pa, poisson)
    uy_plus = _auxiliary_displacement_local(x, y + h, mode, E_Pa, poisson)
    uy_minus = _auxiliary_displacement_local(x, y - h, mode, E_Pa, poisson)
    return np.column_stack(
        [
            (ux_plus - ux_minus) / (2.0 * h),
            (uy_plus - uy_minus) / (2.0 * h),
        ]
    )


def _auxiliary_strain_from_stress(
    sigma: np.ndarray,
    E_Pa: float,
    poisson: float,
) -> np.ndarray:
    E = float(E_Pa)
    nu = float(poisson)
    mu = E / (2.0 * (1.0 + nu))
    sxx = float(sigma[0, 0])
    syy = float(sigma[1, 1])
    sxy = float(sigma[0, 1])
    szz = nu * (sxx + syy)
    exx = (sxx - nu * (syy + szz)) / E
    eyy = (syy - nu * (sxx + szz)) / E
    exy = sxy / (2.0 * mu)
    return np.array([[exx, exy], [exy, eyy]], dtype=float)


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
        # The auxiliary displacement is discontinuous on the negative-x crack
        # face.  Fully damaged crack elements are already excluded; this final
        # guard handles a centroid that lands exactly on the mathematical cut.
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
        nodal_u = np.asarray(u, dtype=float).reshape(-1, 2)[conn]
        grad_global = nodal_u.T @ mesh.dNdx_e[element].T
        grad_actual = rotation.T @ grad_global @ rotation
        sig = sigma_gp[:, element]
        sigma_global = np.array(
            [[sig[0], sig[2]], [sig[2], sig[1]]], dtype=float
        )
        sigma_actual = rotation.T @ sigma_global @ rotation

        sigma_aux = _auxiliary_stress_local(local[0], local[1], mode)
        grad_aux = _auxiliary_gradient_local(
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
    """Return signed ``K_I`` and ``K_II`` for a fixed crack state.

    The domain weight follows the existing J-integral convention: one inside the
    inner contour and zero outside the outer contour.  For a unit auxiliary SIF,
    ``M = 2 K / E'`` under plane strain, so ``K = E' M / 2``.
    """
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
    q_node = _plateau_q(radii, r_inner, r_outer)

    M_I, diag_I = _interaction_for_mode(
        mode="I",
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
    M_II, diag_II = _interaction_for_mode(
        mode="II",
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
    factor = float(mat.Eprime) / 2.0
    K_I = factor * M_I
    K_II = factor * M_II
    return InteractionIntegralResult(
        K_I_Pa_sqrt_m=float(K_I),
        K_II_Pa_sqrt_m=float(K_II),
        M_I_m_per_Pa=float(M_I),
        M_II_m_per_Pa=float(M_II),
        diagnostics={
            "schema": MODEL_ID,
            "plane_strain": True,
            "signed_modes": ["I", "II"],
            "auxiliary_K_Pa_sqrt_m": 1.0,
            "isotropic_auxiliary_fields": True,
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
]
