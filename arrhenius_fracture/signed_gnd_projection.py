"""Signed zero-spin GND projection for the 2-D engineering-strain field.

The bulk constitutive update stores symmetric plastic strain, not plastic spin.
This module therefore exposes the *zero-plastic-spin* Nye estimate explicitly:

    alpha_13 = d(eps_xy^p)/dx - d(eps_xx^p)/dy
    alpha_23 = d(eps_yy^p)/dx - d(eps_xy^p)/dy.

It is a signed, kinematically compatible lower-order observation of the saved
plastic field.  It is not interchangeable with the scalar SSD density.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .incremental_opening_advection import _triangle_rectangle_area


def element_gradients(centers: np.ndarray, values: np.ndarray, *, neighbors: int = 12) -> np.ndarray:
    """Weighted local-linear gradients for element-centred P0 values."""
    xy = np.asarray(centers, float)
    field = np.asarray(values, float)
    if field.ndim == 1:
        field = field[None, :]
    if field.shape[1] != len(xy):
        raise ValueError("values must have one column per element centre")
    k = min(max(int(neighbors), 4), len(xy))
    _, ids = cKDTree(xy).query(xy, k=k)
    out = np.zeros((field.shape[0], len(xy), 2), float)
    for e in range(len(xy)):
        jj = np.atleast_1d(ids[e])
        delta = xy[jj] - xy[e]
        scale = max(float(np.max(np.linalg.norm(delta, axis=1))), 1.0e-30)
        A = np.column_stack((delta[:, 0] / scale, delta[:, 1] / scale,
                             np.ones(len(jj))))
        distance = np.linalg.norm(delta, axis=1)
        w = 1.0 / np.maximum(distance / scale, 0.15)
        Aw = A * w[:, None]
        for q in range(field.shape[0]):
            coef, *_ = np.linalg.lstsq(Aw, field[q, jj] * w, rcond=None)
            out[q, e] = coef[:2] / scale
    return out


def signed_gnd_density(nodes: np.ndarray, elems: np.ndarray, ep_gp: np.ndarray,
                       burgers_m: float, *, slip_angles_deg=(45.0, -45.0)) -> dict:
    """Return signed GND densities projected onto two in-plane Burgers axes."""
    triangles = np.asarray(nodes, float)[np.asarray(elems, int)]
    centers = triangles.mean(axis=1)
    ep = np.asarray(ep_gp, float)
    if ep.shape != (3, len(triangles)):
        raise ValueError("ep_gp must use (exx, eyy, engineering gamma_xy)")
    grad = element_gradients(centers, ep)
    # The stored third component is engineering shear gamma_xy=2 eps_xy.
    depsxy_dx = 0.5 * grad[2, :, 0]
    depsxy_dy = 0.5 * grad[2, :, 1]
    alpha = np.column_stack((depsxy_dx - grad[0, :, 1],
                             grad[1, :, 0] - depsxy_dy))
    angles = np.deg2rad(np.asarray(slip_angles_deg, float))
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    rho_signed = directions @ alpha.T / max(abs(float(burgers_m)), 1.0e-30)
    return {"centers": centers, "triangles": triangles, "alpha_per_m": alpha,
            "directions": directions, "rho_signed_m2": rho_signed,
            "assumption": "zero_plastic_spin_symmetric_distortion"}


def signed_profiles(*, triangles: np.ndarray, rho_signed_m2: np.ndarray,
                    tip_x_m: float, active_length_m: float, active_bins: int,
                    wake_length_m: float, wake_bins: int,
                    capture_width_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact triangle-overlap integration into active and trailing-wake bins."""
    tri = np.asarray(triangles, float)
    rho = np.asarray(rho_signed_m2, float)
    half = 0.5 * float(capture_width_m)
    active = np.zeros((rho.shape[0], int(active_bins)))
    wake = np.zeros((rho.shape[0], int(wake_bins)))
    adx = float(active_length_m) / active.shape[1]
    wdx = float(wake_length_m) / wake.shape[1]
    for e in range(len(tri)):
        xmin, xmax = np.min(tri[e, :, 0]), np.max(tri[e, :, 0])
        ymin, ymax = np.min(tri[e, :, 1]), np.max(tri[e, :, 1])
        if ymax < -half or ymin > half:
            continue
        alo = max(int(np.floor((xmin - tip_x_m) / adx)), 0)
        ahi = min(int(np.floor((xmax - tip_x_m) / adx)), active.shape[1] - 1)
        for j in range(alo, ahi + 1):
            area = _triangle_rectangle_area(
                tri[e], tip_x_m + j * adx, tip_x_m + (j + 1) * adx, -half, half)
            active[:, j] += rho[:, e] * area
        # wake coordinate is distance behind the current tip.
        wlo = max(int(np.floor((tip_x_m - xmax) / wdx)), 0)
        whi = min(int(np.floor((tip_x_m - xmin) / wdx)), wake.shape[1] - 1)
        for j in range(wlo, whi + 1):
            area = _triangle_rectangle_area(
                tri[e], tip_x_m - (j + 1) * wdx, tip_x_m - j * wdx, -half, half)
            wake[:, j] += rho[:, e] * area
    return active, wake


def swept_tip_content(*, triangles: np.ndarray, rho_signed_m2: np.ndarray,
                      directions: np.ndarray, old_tip_x_m: float,
                      new_tip_x_m: float, capture_width_m: float) -> dict:
    """Integrate signed GND content consumed by a moving crack-tip strip.

    The two columns in ``signed_count_by_system_face`` are the lower and upper
    crack faces.  ``opening_count_by_system_face`` retains only Burgers motion
    whose y projection opens the corresponding face.  This is a signed
    crack-swept transaction, not a claim that the bulk FE model resolves a
    time-continuous mobile-dislocation flux into the tip.
    """
    tri = np.asarray(triangles, dtype=float)
    rho = np.asarray(rho_signed_m2, dtype=float)
    dirs = np.asarray(directions, dtype=float)
    if rho.ndim != 2 or rho.shape[1] != len(tri):
        raise ValueError("rho_signed_m2 must have shape (systems, elements)")
    if dirs.shape != (rho.shape[0], 2):
        raise ValueError("directions must have shape (systems, 2)")
    xmin = min(float(old_tip_x_m), float(new_tip_x_m))
    xmax = max(float(old_tip_x_m), float(new_tip_x_m))
    half = 0.5 * max(float(capture_width_m), 0.0)
    signed = np.zeros((rho.shape[0], 2), dtype=float)
    if xmax > xmin and half > 0.0:
        bbox = ((np.max(tri[:, :, 0], axis=1) >= xmin)
                & (np.min(tri[:, :, 0], axis=1) <= xmax)
                & (np.max(tri[:, :, 1], axis=1) >= -half)
                & (np.min(tri[:, :, 1], axis=1) <= half))
        for e in np.flatnonzero(bbox):
            lower_area = _triangle_rectangle_area(
                tri[e], xmin, xmax, -half, 0.0)
            upper_area = _triangle_rectangle_area(
                tri[e], xmin, xmax, 0.0, half)
            signed[:, 0] += rho[:, e] * lower_area
            signed[:, 1] += rho[:, e] * upper_area
    # Outward face normals: lower=-y, upper=+y.  A signed Burgers step opens a
    # face when n_y * b_y is positive.  Counts remain dimensionless line/event
    # counts because rho [m^-2] is integrated over area [m^2].
    face_normal_y = np.array((-1.0, 1.0), dtype=float)
    projected = signed * dirs[:, 1, None] * face_normal_y[None, :]
    opening = np.maximum(projected, 0.0)
    return {
        "signed_count_by_system_face": signed,
        "normal_projected_count_by_system_face": projected,
        "opening_count_by_system_face": opening,
        "signed_count_total": float(np.sum(signed)),
        "absolute_count_total": float(np.sum(np.abs(signed))),
        "opening_effective_count": float(np.sum(opening)),
        "face_order": ("lower", "upper"),
        "scope": "crack_swept_signed_GND_not_continuous_mobile_flux",
    }


def shielding_from_profiles(active: np.ndarray, wake: np.ndarray, *, active_x_m: np.ndarray,
                            wake_x_m: np.ndarray, orientation_factors: np.ndarray,
                            G_Pa: float, nu: float, burgers_m: float,
                            core_m: float, wake_projection: float = 1.0) -> dict:
    factors = np.asarray(orientation_factors, float)[:, None]
    prefactor = float(G_Pa) * float(burgers_m) / max(1.0 - float(nu), 1.0e-6)
    ka = prefactor / np.sqrt(2.0 * np.pi * np.maximum(active_x_m, core_m))
    kw = prefactor / np.sqrt(2.0 * np.pi * np.maximum(wake_x_m, core_m))
    active_K = float(np.sum(factors * active * ka[None, :]))
    wake_K = float(wake_projection * np.sum(factors * wake * kw[None, :]))
    return {"active_K_Pa_sqrt_m": active_K, "wake_K_Pa_sqrt_m": wake_K,
            "total_K_Pa_sqrt_m": active_K + wake_K}

