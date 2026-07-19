"""
Finite element assembly and mechanics solve for plane-strain triangular elements.

Key fixes from original:
1. Vectorized assembly (much faster than element-by-element loops)
2. Proper plane-strain sigma_zz in von Mises computation
3. Clean separation of elastic and plastic contributions
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Tuple
from .mesh import TriMesh, BoundaryData
from .config import ElasticProperties


def plane_strain_D(mat: ElasticProperties) -> np.ndarray:
    """
    Plane-strain elasticity matrix D (3x3).
    Maps [eps_xx, eps_yy, gamma_xy] -> [sig_xx, sig_yy, sig_xy].
    """
    E, nu = mat.E, mat.nu
    c = E / ((1 + nu) * (1 - 2 * nu))
    return c * np.array([
        [1 - nu, nu, 0],
        [nu, 1 - nu, 0],
        [0, 0, (1 - 2 * nu) / 2]
    ])


def assemble_mechanics(
    mesh: TriMesh, u: np.ndarray, ep_gp: np.ndarray,
    rho_gp: np.ndarray, d: np.ndarray,
    D: np.ndarray, mat: ElasticProperties,
    kappa: float = 1e-6,
    cohesive_network=None,
) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """
    Assemble global stiffness and internal force vector (vectorized).

    Returns
    -------
    K : (ndof, ndof) sparse stiffness matrix
    Rint : (ndof,) internal force vector
    sigma_gp : (3, ne) stress at each Gauss point
    sigma_eq_gp : (ne,) von Mises equivalent stress (PROPER plane strain)
    sigma1_gp : (ne,) maximum principal stress
    psi_e_gp : (ne,) positive elastic energy density at each GP
    """
    ne = mesh.ne
    nn = mesh.nn
    ndof = mesh.ndof
    nu = mat.nu

    conn = mesh.elems  # (ne, 3)

    # Build element DOF indices (ne, 6)
    edofs = np.zeros((ne, 6), dtype=int)
    edofs[:, 0] = 2 * conn[:, 0]
    edofs[:, 1] = 2 * conn[:, 0] + 1
    edofs[:, 2] = 2 * conn[:, 1]
    edofs[:, 3] = 2 * conn[:, 1] + 1
    edofs[:, 4] = 2 * conn[:, 2]
    edofs[:, 5] = 2 * conn[:, 2] + 1

    # Gather element displacements (ne, 6)
    ue = u[edofs]

    # Total strain: eps = B @ ue, vectorized as einsum
    # B_e is (ne, 3, 6), ue is (ne, 6)
    eps_tot = np.einsum('eij,ej->ei', mesh.B_e, ue)  # (ne, 3)

    # Elastic strain
    eps_e = eps_tot - ep_gp.T  # (ne, 3), ep_gp is (3, ne)

    # Broken-material stiffness degradation per element
    dgp = np.mean(d[conn], axis=1)  # (ne,)
    g_d = (1 - dgp)**2 + kappa      # (ne,)

    # Stress: sig = g_d * D @ eps_e
    sig = g_d[:, None] * (eps_e @ D.T)  # (ne, 3)
    sigma_gp = sig.T  # (3, ne)

    # Proper plane-strain von Mises
    sx = sig[:, 0]
    sy = sig[:, 1]
    txy = sig[:, 2]
    szz = nu * (sx + sy)

    sigma_eq_gp = np.sqrt(0.5 * ((sx-sy)**2 + (sy-szz)**2 + (szz-sx)**2)
                          + 3 * txy**2)

    # Maximum in-plane principal stress
    s_avg = 0.5 * (sx + sy)
    R = np.sqrt((0.5*(sx-sy))**2 + txy**2)
    sigma1_gp = s_avg + R

    # Positive elastic energy density, DEGRADED by g_d to match the stress and
    # stiffness (which are both scaled by g_d).  Without this factor the energy
    # density in stiffness-killed elements (g_d->0, eps_e large) is spuriously
    # full, breaking the global energy balance (sum psi*A != 1/2 u^T K u).
    psi_e_gp = g_d * 0.5 * np.sum(eps_e * (eps_e @ D.T), axis=1)  # (ne,)
    psi_e_gp = np.where(sigma1_gp > 0, psi_e_gp, 0.0)

    # --- Sparse stiffness assembly ---
    # Ke = B^T @ (g_d * D) @ B * A  for each element
    # gD = g_d[:, None, None] * D[None, :, :]  # (ne, 3, 3)
    gD = g_d[:, None, None] * D[None, :, :]
    A = mesh.area_e  # (ne,)

    # Ke_all = B^T @ gD @ B * A  ->  (ne, 6, 6)
    BtgD = np.einsum('eji,ejk->eik', mesh.B_e, gD)  # (ne, 6, 3)
    Ke_all = np.einsum('eij,ejk->eik', BtgD, mesh.B_e) * A[:, None, None]

    # Internal force: Re = B^T @ sig * A  ->  (ne, 6)
    Re_all = np.einsum('eji,ej->ei', mesh.B_e, sig) * A[:, None]

    # Build sparse triplets
    ii = edofs[:, :, None].repeat(6, axis=2)  # (ne, 6, 6)
    jj = edofs[:, None, :].repeat(6, axis=1)  # (ne, 6, 6)

    K = sparse.csr_matrix(
        (Ke_all.ravel(), (ii.ravel(), jj.ravel())),
        shape=(ndof, ndof)
    )

    # Scatter internal force
    Rint = np.zeros(ndof)
    np.add.at(Rint, edofs.ravel(), Re_all.ravel())

    # Optional zero-thickness Arrhenius cohesive interfaces.  Their irreversible
    # state is controlled outside this routine by the hazard backend; assembly
    # only contributes the traction response of whatever intact fraction remains.
    if cohesive_network is not None:
        from .cohesive import cohesive_contribution
        Kcz, Rcz = cohesive_contribution(cohesive_network, u, ndof)
        K = K + Kcz
        Rint = Rint + Rcz

    return K, Rint, sigma_gp, sigma_eq_gp, sigma1_gp, psi_e_gp


def stress_state(mesh: TriMesh, u: np.ndarray, ep_gp: np.ndarray,
                 d: np.ndarray, D: np.ndarray, mat: ElasticProperties,
                 kappa: float = 1e-6):
    """Recompute (sigma_gp, sigma_eq_gp, sigma1_gp, psi_e_gp) at the CURRENT
    displacement without assembling K.  Used after the mechanics solve so the
    plastic update, event drive, and equilibrium solve all see the
    EQUILIBRIUM stress rather than the pre-solve stale field."""
    conn = mesh.elems
    edofs = np.zeros((mesh.ne, 6), dtype=int)
    edofs[:, 0] = 2*conn[:, 0]; edofs[:, 1] = 2*conn[:, 0]+1
    edofs[:, 2] = 2*conn[:, 1]; edofs[:, 3] = 2*conn[:, 1]+1
    edofs[:, 4] = 2*conn[:, 2]; edofs[:, 5] = 2*conn[:, 2]+1
    ue = u[edofs]
    eps_tot = np.einsum('eij,ej->ei', mesh.B_e, ue)
    eps_e = eps_tot - ep_gp.T
    dgp = np.mean(d[conn], axis=1)
    g_d = (1 - dgp)**2 + kappa
    sig = g_d[:, None] * (eps_e @ D.T)
    sx, sy, txy = sig[:, 0], sig[:, 1], sig[:, 2]
    szz = mat.nu * (sx + sy)
    sigma_eq = np.sqrt(0.5*((sx-sy)**2 + (sy-szz)**2 + (szz-sx)**2) + 3*txy**2)
    s_avg = 0.5*(sx+sy); R = np.sqrt((0.5*(sx-sy))**2 + txy**2)
    sigma1 = s_avg + R
    psi = g_d * 0.5 * np.sum(eps_e * (eps_e @ D.T), axis=1)
    psi = np.where(sigma1 > 0, psi, 0.0)
    return sig.T, sigma_eq, sigma1, psi


def solve_dirichlet(
    K: sparse.csr_matrix, Rint: np.ndarray, u: np.ndarray,
    bnd: BoundaryData, Uy_top: float, Uy_bot: float
) -> Tuple[np.ndarray, float]:
    """
    Solve mechanics with Dirichlet BCs (symmetric opening).

    Returns updated displacement and top reaction force.
    """
    nn = len(u) // 2
    ndof = 2 * nn

    prescribed = np.zeros(ndof, dtype=bool)
    u_pres = np.zeros(ndof)

    # Top: fix y-displacement
    prescribed[2 * bnd.top_nodes + 1] = True
    u_pres[2 * bnd.top_nodes + 1] = Uy_top

    # Bottom: fix y-displacement
    prescribed[2 * bnd.bot_nodes + 1] = True
    u_pres[2 * bnd.bot_nodes + 1] = Uy_bot

    # Left-bottom corner: fix x and y
    prescribed[2 * bnd.left_bot] = True
    u_pres[2 * bnd.left_bot] = 0
    prescribed[2 * bnd.left_bot + 1] = True
    u_pres[2 * bnd.left_bot + 1] = Uy_bot

    # Right-bottom corner: fix x only
    prescribed[2 * bnd.right_bot] = True
    u_pres[2 * bnd.right_bot] = 0

    free = ~prescribed

    # INCREMENTAL solve: K du = -Rint with du_pres = u_pres - u_old.
    # The previous ABSOLUTE form (rhs = -Rint - K_fp u_pres, solving for
    # u_new directly) is exact only when u_old is itself an elastic
    # equilibrium: the -u_old and BC-lift terms then cancel.  With a plastic
    # eigenstrain force inside Rint the cancellation breaks and the solve
    # returns u* MINUS the previous plastic lift (u_new = u* - K^-1 f_p_old):
    # the body never reaches equilibrium, the staggered plasticity ratchets
    # against the residual stress, and plastic work is booked every ratchet
    # (the high-T Wp/Wext ~ 1e5% runaway).  The incremental form is exact
    # for the linear problem from ANY starting state.
    K_csr = K.tocsr()
    du_pres = u_pres[prescribed] - u[prescribed]
    rhs = -Rint[free] - K_csr[np.ix_(free, prescribed)] @ du_pres

    u_new = u.copy()
    u_new[free] = u[free] + spsolve(K_csr[np.ix_(free, free)], rhs)
    u_new[prescribed] = u_pres[prescribed]

    # Reaction force on top nodes (linear update of the internal force)
    Rfull = Rint + K_csr @ (u_new - u)
    Ftop = np.sum(Rfull[2 * bnd.top_nodes + 1])

    return u_new, Ftop



def boundary_reaction_forces(K: sparse.csr_matrix, Rint: np.ndarray, u: np.ndarray,
                             bnd: BoundaryData) -> Tuple[float, float, float]:
    """Return top, bottom, and total absolute boundary reactions.

    The mechanics solve uses displacement control on both top and bottom
    boundaries.  A scalar top reaction alone is a useful load metric, but it is
    not sufficient for a thermodynamic work audit because the bottom boundary
    also moves.  This helper returns the signed reactions on both moving
    boundaries using the same residual convention as ``solve_dirichlet``.

    Returns
    -------
    Ftop, Fbot, Fabs : floats
        Signed top/bottom y-reactions and sum of absolute magnitudes.  Units are
        force per out-of-plane thickness for the 2-D model.
    """
    K_csr = K.tocsr()
    Rfull = K_csr @ u + Rint
    Ftop = float(np.sum(Rfull[2 * bnd.top_nodes + 1]))
    Fbot = float(np.sum(Rfull[2 * bnd.bot_nodes + 1]))
    Fabs = float(abs(Ftop) + abs(Fbot))
    return Ftop, Fbot, Fabs


def elastic_energy_densities(mesh: TriMesh, u: np.ndarray, ep_gp: np.ndarray,
                             sigma_gp: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return stored and undegraded elastic energy densities.

    ``assemble_mechanics`` returns ``psi_e_gp`` as the positive undegraded
    tensile energy diagnostic.  That is not the stored mechanical
    energy of the degraded body.  For a thermodynamic audit the stored energy
    must use the stress that actually appears in the equilibrium solve:

        psi_store = 1/2 eps_e : sigma_degraded.

    This helper also returns the undegraded elastic energy density so the two
    quantities can be compared in diagnostics.
    """
    conn = mesh.elems
    edofs = np.zeros((mesh.ne, 6), dtype=int)
    edofs[:, 0] = 2 * conn[:, 0]
    edofs[:, 1] = 2 * conn[:, 0] + 1
    edofs[:, 2] = 2 * conn[:, 1]
    edofs[:, 3] = 2 * conn[:, 1] + 1
    edofs[:, 4] = 2 * conn[:, 2]
    edofs[:, 5] = 2 * conn[:, 2] + 1
    ue = u[edofs]
    eps_tot = np.einsum('eij,ej->ei', mesh.B_e, ue)
    eps_e = eps_tot - ep_gp.T
    sig = sigma_gp.T
    psi_stored = 0.5 * np.sum(eps_e * sig, axis=1)
    psi_stored = np.maximum(psi_stored, 0.0)
    psi_undegraded = 0.5 * np.sum(eps_e * (eps_e @ D.T), axis=1)
    psi_undegraded = np.maximum(psi_undegraded, 0.0)
    return psi_stored, psi_undegraded

def project_gp_to_nodes(mesh: TriMesh, val_gp: np.ndarray) -> np.ndarray:
    """
    Project Gauss-point values to nodes using area-weighted averaging.

    Parameters
    ----------
    val_gp : (ne,) or (3, ne) array of GP values

    Returns
    -------
    val_node : (nn,) or (3, nn) nodal values
    """
    if val_gp.ndim == 1:
        return _project_scalar(mesh, val_gp)
    else:
        result = np.zeros((val_gp.shape[0], mesh.nn))
        for i in range(val_gp.shape[0]):
            result[i] = _project_scalar(mesh, val_gp[i])
        return result


def _project_scalar(mesh: TriMesh, val_gp: np.ndarray) -> np.ndarray:
    """Project scalar GP values to nodes (vectorized)."""
    conn = mesh.elems  # (ne, 3)
    A = mesh.area_e    # (ne,)
    w = A / 3          # (ne,) weight per node contribution

    # Each node gets weighted contributions from its connected elements
    acc = np.zeros(mesh.nn)
    wacc = np.zeros(mesh.nn)

    contrib = (val_gp * w / 3)  # 1/3 shape function at centroid * (A/3 weight)
    # Actually: N_a = 1/3 at centroid, so contribution = (1/3)*val_gp*(A)
    contrib = val_gp * A / 3

    np.add.at(acc, conn[:, 0], contrib)
    np.add.at(acc, conn[:, 1], contrib)
    np.add.at(acc, conn[:, 2], contrib)

    w_contrib = A / 3
    np.add.at(wacc, conn[:, 0], w_contrib)
    np.add.at(wacc, conn[:, 1], w_contrib)
    np.add.at(wacc, conn[:, 2], w_contrib)

    return acc / np.maximum(wacc, 1e-30)

