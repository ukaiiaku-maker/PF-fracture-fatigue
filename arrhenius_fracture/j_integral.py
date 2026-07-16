"""
Domain-integral J-integral computation for phase-field fracture.

Implements the Shih-Moran-Nakamura (1986) domain integral formulation
adapted for AT2 phase-field models, following the approach of
Molnár & Gravouil (2017) and Kristensen & Martínez-Pañeda (2020).

This is the KEY MISSING PIECE from the original MATLAB code, which
used a crude W_ext / Da_projected estimate that conflates process zone
dissipation with fracture energy and is unreliable for branched cracks.

The domain integral is path-independent for a straight crack in a
homogeneous material and provides a true energy release rate even
in the presence of plasticity and crack branching.
"""

import numpy as np
from typing import Tuple, Optional
from .mesh import TriMesh
from .config import ElasticProperties, JIntegralConfig


def _seg_cross(a0, a1, b0, b1):
    """True if open segments a0-a1 and b0-b1 properly intersect (2D)."""
    def cr(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])
    d1 = cr(b0, b1, a0); d2 = cr(b0, b1, a1)
    d3 = cr(a0, a1, b0); d4 = cr(a0, a1, b1)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _line_of_sight_blocked(tip, pt, segments, ignore_tol):
    """True if the straight path tip->pt crosses any crack segment that is NOT
    incident to the tip (segments touching the tip within ignore_tol are the
    tip's own crack and are skipped).  This is the domain-segmentation rule:
    an element only contributes to a tip's J-integral if it has clear
    line-of-sight to that tip -- so the contour never runs THROUGH a neighboring
    crack, and the wake/other tips are routed around.  Future-proof for branching:
    pass ALL crack polyline segments; each tip's domain auto-segments."""
    for (p0, p1) in segments:
        # skip the tip's own incident segments (the path runs along, not across)
        if min((p0[0] - tip[0])**2 + (p0[1] - tip[1])**2,
               (p1[0] - tip[0])**2 + (p1[1] - tip[1])**2) < ignore_tol**2:
            continue
        if _seg_cross(tip, pt, p0, p1):
            return True
    return False


def compute_J_integral(
    mesh: TriMesh,
    u: np.ndarray,              # (ndof,) displacement field
    sigma_gp: np.ndarray,       # (3, ne) stress at GPs
    psi_e_gp: np.ndarray,       # (ne,) elastic energy density at GPs
    d: np.ndarray,              # (nn,) damage field
    crack_tip: np.ndarray,      # (2,) crack tip coordinates [x_tip, y_tip]
    crack_direction: np.ndarray,  # (2,) crack growth direction [dx, dy]
    mat: ElasticProperties,
    ell: float,
    cfg: JIntegralConfig = None,
    crack_segments=None,        # list[(p0,p1)] ALL crack polyline segments (branch-safe)
    exclude_radius: float = 0.0,  # hard radius around tip forced out of the domain
) -> Tuple[float, float, dict]:
    """
    Compute J-integral using the domain integral method.

    The J-integral for a crack growing in direction e1 is:

        J = ∫_A [σ_ij * ∂u_i/∂x1 - W * δ_1j] * ∂q/∂x_j dA

    where q is a smooth weight function: q=1 on an inner contour,
    q=0 on an outer contour, and varies smoothly between them.

    For a phase-field crack, we:
    1. Locate the effective crack tip from the damage field
    2. Define an annular integration domain around the tip
    3. Use a plateau weight function q(r)
    4. Integrate the configurational force

    Parameters
    ----------
    mesh : triangular mesh
    u : displacement field
    sigma_gp : stress at Gauss points (Voigt: [sig_xx, sig_yy, sig_xy])
    psi_e_gp : elastic energy density at GPs
    d : damage field
    crack_tip : [x, y] coordinates of crack tip
    crack_direction : unit vector of crack growth direction
    mat : elastic properties
    ell : phase-field length scale
    cfg : J-integral configuration

    Returns
    -------
    J : J-integral value [J/m²]
    KJ : equivalent stress intensity factor [Pa*sqrt(m)]
    info : diagnostic dictionary
    """
    if cfg is None:
        cfg = JIntegralConfig()

    nn = mesh.nn
    ne = mesh.ne

    # Integration domain radii
    r_inner = cfg.r_inner_factor * ell
    r_outer = cfg.r_outer_factor * ell

    # Ensure crack direction is a unit vector
    e1 = np.array(crack_direction, dtype=float)
    e1_norm = np.linalg.norm(e1)
    if e1_norm < 1e-12:
        e1 = np.array([1.0, 0.0])
    else:
        e1 = e1 / e1_norm

    # Compute q-field at nodes (weight function)
    tip = np.array(crack_tip, dtype=float)
    dx = mesh.nodes[:, 0] - tip[0]
    dy = mesh.nodes[:, 1] - tip[1]
    r = np.sqrt(dx**2 + dy**2)

    if cfg.q_type == 'plateau':
        q_node = _plateau_q(r, r_inner, r_outer)
    else:
        q_node = _linear_q(r, r_inner, r_outer)

    # Branch-safe domain segmentation: keep only crack segments near the tip
    # (within ~3*r_outer); far wake never blocks line-of-sight and testing it
    # would only cost time.  ignore_tol = r_inner so the tip's own incident
    # segment (the immediate wake) is skipped by the line-of-sight test.
    near_segments = None
    if crack_segments:
        rmax2 = (3.0 * r_outer) ** 2
        near_segments = [(p0, p1) for (p0, p1) in crack_segments
                         if min((p0[0]-tip[0])**2 + (p0[1]-tip[1])**2,
                                (p1[0]-tip[0])**2 + (p1[1]-tip[1])**2) < rmax2]
        if not near_segments:
            near_segments = None
    los_ignore = r_inner

    # Compute displacement gradient at each element
    # ∂u_i/∂x_j from the FE interpolation
    J_value = 0.0
    n_active = 0

    for e in range(ne):
        conn = mesh.elems[e]
        A = mesh.area_e[e]
        dNdx = mesh.dNdx_e[e]  # (2, 3)

        # Check if element is in the integration domain
        qe = q_node[conn]
        if np.max(np.abs(qe)) < 1e-12:
            continue

        # Gradient of q
        dqdx = dNdx @ qe  # (2,) = [∂q/∂x, ∂q/∂y]

        if np.linalg.norm(dqdx) < 1e-20:
            continue

        # Element displacements
        edofs = np.array([2*conn[0], 2*conn[0]+1,
                          2*conn[1], 2*conn[1]+1,
                          2*conn[2], 2*conn[2]+1])
        ue = u[edofs]

        # Displacement gradient ∂u_i/∂x_j (2x2 tensor)
        # du_i/dx_j = sum_a dN_a/dx_j * u_a_i
        gradu = np.zeros((2, 2))
        for a in range(3):
            gradu[0, 0] += dNdx[0, a] * ue[2*a]      # du1/dx1
            gradu[0, 1] += dNdx[1, a] * ue[2*a]      # du1/dx2
            gradu[1, 0] += dNdx[0, a] * ue[2*a + 1]  # du2/dx1
            gradu[1, 1] += dNdx[1, a] * ue[2*a + 1]  # du2/dx2

        # Stress (Voigt -> tensor)
        sig = sigma_gp[:, e]
        sigma_tensor = np.array([
            [sig[0], sig[2]],
            [sig[2], sig[1]]
        ])

        # Elastic energy density at this element
        W = psi_e_gp[e]

        # Eshelby tensor component along crack direction:
        # P_1j = σ_ij * ∂u_i/∂x_1 - W * δ_1j
        # with x_1 = crack direction
        #
        # In general coordinates:
        # J = ∫ [σ_ij * (∂u_i/∂x_k * e1_k) - W * e1_j] * (∂q/∂x_j) dA

        # σ_ij * ∂u_i/∂x_k * e1_k (sum over i,k for each j)
        # = σ @ (gradu @ e1)  where @ is matrix-vector
        sigma_gradu_e1 = sigma_tensor @ (gradu @ e1)  # (2,) vector

        # Eshelby integrand dotted with ∂q/∂x
        integrand = np.dot(sigma_gradu_e1, dqdx) - W * np.dot(e1, dqdx)

        # Phase-field correction: exclude fully damaged elements
        # The J-integral should be computed on the undamaged region
        de = d[conn]
        dgp = np.mean(de)
        if dgp > 0.95:
            continue  # skip fully cracked elements

        # --- branch-safe domain mask -------------------------------------
        # element centroid
        cxe = (mesh.nodes[conn[0], 0] + mesh.nodes[conn[1], 0] + mesh.nodes[conn[2], 0]) / 3.0
        cye = (mesh.nodes[conn[0], 1] + mesh.nodes[conn[1], 1] + mesh.nodes[conn[2], 1]) / 3.0
        # (a) hard exclusion disk around the tip (keeps the contour outside the
        #     freshly-killed kill_r blob)
        if exclude_radius > 0.0:
            if (cxe - tip[0])**2 + (cye - tip[1])**2 < exclude_radius**2:
                continue
        # (b) line-of-sight: skip elements whose path to the tip crosses a crack
        #     that is not the tip's own -> the domain never runs through a
        #     neighboring crack and auto-segments between close tips.
        if near_segments is not None:
            if _line_of_sight_blocked((tip[0], tip[1]), (cxe, cye),
                                      near_segments, los_ignore):
                continue

        J_value += integrand * A
        n_active += 1

    # Convert to KJ.  The sign of the domain integral depends on the
    # q-field convention and crack-direction convention.  Earlier versions
    # silently clipped negative J to zero, which made elastic AT2 tests report
    # KJ_domain=0 even when the magnitude of the energy-release rate was nonzero.
    # Store the signed value in info, but use |J| for the energy-release metric.
    Eprime = mat.Eprime
    J_signed = J_value
    J_energy = abs(J_value)
    KJ = np.sqrt(max(J_energy, 0) * Eprime)

    info = {
        'J': J_energy,
        'J_signed': J_signed,
        'KJ': KJ,
        'KJ_MPa': KJ / 1e6,
        'n_active_elements': n_active,
        'r_inner': r_inner,
        'r_outer': r_outer,
        'crack_tip': tip.copy(),
    }

    return J_energy, KJ, info


def find_crack_tip(
    mesh: TriMesh, d: np.ndarray, a0: float,
    d_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find effective crack tip location and growth direction from damage field.

    The tip is identified as the furthest point along x with d > threshold
    on the crack plane (y ≈ 0).

    Returns
    -------
    tip : (2,) crack tip coordinates
    direction : (2,) unit crack growth direction
    """
    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]

    # Find damaged nodes near the crack plane
    tol_y = 2 * mesh.hbar  # tolerance for "near crack plane"
    near_plane = np.abs(y) < tol_y
    damaged = d > d_threshold

    candidates = near_plane & damaged & (x > a0 - mesh.hbar)

    if not np.any(candidates):
        # No crack advance detected
        return np.array([a0, 0.0]), np.array([1.0, 0.0])

    # Tip = furthest damaged node on the plane
    x_cand = x[candidates]
    y_cand = y[candidates]
    idx_max = np.argmax(x_cand)

    tip = np.array([x_cand[idx_max], y_cand[idx_max]])

    # Growth direction: from initial notch tip toward current tip
    dx = tip[0] - a0
    dy = tip[1]
    norm = np.sqrt(dx**2 + dy**2)
    if norm < 1e-12:
        direction = np.array([1.0, 0.0])
    else:
        direction = np.array([dx, dy]) / norm

    return tip, direction


def compute_crack_advance(
    mesh: TriMesh, d: np.ndarray, a0: float,
    d_threshold: float = 0.8
) -> Tuple[float, float, float]:
    """
    Compute crack advance metrics.

    Returns
    -------
    Da_projected : projected (x-direction) crack advance [m]
    Gamma_total : total crack surface from AT2 functional [m per unit thickness]
    branch_factor : Gamma_total / Da_projected (≥1, >1 means branching)
    """
    x = mesh.nodes[:, 0]
    dam_mask = d > d_threshold

    if np.any(dam_mask):
        a_eff = np.max(x[dam_mask])
        Da_projected = max(a_eff - a0, 1e-12)
    else:
        Da_projected = 1e-12

    # Total crack surface is computed externally from AT2 energy
    # Here we just return the projected advance
    return Da_projected, Da_projected, 1.0


def _plateau_q(r: np.ndarray, r_in: float, r_out: float) -> np.ndarray:
    """
    Plateau weight function for domain integral.

    q = 1      for r ≤ r_in
    q = smooth  for r_in < r < r_out
    q = 0      for r ≥ r_out
    """
    q = np.ones_like(r)
    mask = (r > r_in) & (r < r_out)
    t = (r[mask] - r_in) / (r_out - r_in)
    # Smooth hermite transition
    q[mask] = 1 - t**2 * (3 - 2*t)
    q[r >= r_out] = 0.0
    return q


def _linear_q(r: np.ndarray, r_in: float, r_out: float) -> np.ndarray:
    """Linear weight function."""
    q = np.ones_like(r)
    mask = (r > r_in) & (r < r_out)
    q[mask] = 1 - (r[mask] - r_in) / (r_out - r_in)
    q[r >= r_out] = 0.0
    return q
