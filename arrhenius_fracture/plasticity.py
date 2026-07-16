"""
Arrhenius-Taylor plasticity following the Arrhenius Mechanics formulation
(Mirzaei & Dillon).

The flow stress for dislocation segment depinning is (eq. 17):

    sigma = (2b*sqrt(rho) / v*) * [A + kT*ln(eps_dot / (16*rho^2*b^4))]

where:
    phi = delta/b = 1/(2b*sqrt(rho))  — stress concentration at pinning nodes
    Xi  = v*/delta^3                  — volume fraction of active sites
    A   = H* - T*S* + k*ln(eta_0)    — grouped barrier (isothermal fitting param)

This naturally produces:
    - Taylor hardening:  sigma ~ sqrt(rho) at moderate rho
    - Peak stress:       at rho where A + kT*ln(...) = 4kT
    - Strain softening:  at rho beyond peak (log term dominates)

The softening onset depends on T through A/(kT): higher T = earlier softening.
No artificial rho_cap or Taylor back-stress needed.

The rate form (used for the explicit update) is:

    eps_dot = eta_0 * (b*v*/delta^4) * exp(-(H* - T*S* - sigma*phi*v*) / (kT))

where sigma*phi*v* = sigma * (delta/b) * v* is the stress-volume work at the
LOCAL stress (amplified by the dislocation stress concentration).
"""

import numpy as np
from typing import Tuple
try:
    from scipy.special import gammaincinv as _gammaincinv
except Exception:  # pragma: no cover
    _gammaincinv = None
from .config import (
    KB, EV_TO_J, ElasticProperties, PlasticityBarrier, DislocationConfig
)
from .materials import PlasticityModel




def _von_mises_plane_strain(sigma_gp: np.ndarray, nu: float):
    """Return plane-strain von Mises equivalent stress and deviatoric norm."""
    sx = sigma_gp[0, :]
    sy = sigma_gp[1, :]
    txy = sigma_gp[2, :]
    szz = nu * (sx + sy)
    p = (sx + sy + szz) / 3.0
    sd_xx = sx - p
    sd_yy = sy - p
    sd_zz = szz - p
    norm_s = np.sqrt(sd_xx**2 + sd_yy**2 + sd_zz**2 + 2.0*txy**2)
    seq = np.sqrt(1.5) * norm_s
    return seq, norm_s, (sd_xx, sd_yy, txy)


def peierls_flow_stress(T, eps_ref, plast_model, disl_cfg, b):
    """Lattice-friction (Peierls / kink-pair) flow-stress branch, phi_P = 1.

    A genuine thermally activated branch (paper Eq. 16, geometry-independent):
        eps_ref = eta0 * exp(-(H_P - T*S_P - sigma_P*v_P)/kT)
    inverted for the resolved flow stress
        sigma_P = max( (H_P - T*S_P) + kT*ln(eps_ref/eta0), 0 ) / v_P.

    This branch does NOT couple to the dislocation density rho; it sets the
    low-temperature thermal stress and collapses as T rises (A_P -> 0).
    Returns a scalar stress [Pa].
    """
    if not bool(getattr(disl_cfg, 'use_peierls_floor', False)):
        return 0.0
    H_P = float(getattr(disl_cfg, 'peierls_H0_eV', 1.7)) * EV_TO_J
    v_P = max(float(getattr(disl_cfg, 'peierls_v0_b3', 5.0)) * b**3, 1e-36)
    S_P = float(getattr(disl_cfg, 'peierls_S_kB', 0.0)) * KB
    A_P = H_P - T * S_P + KB * T * np.log(max(eps_ref, 1e-300) / max(plast_model.p.eta0, 1e-300))
    sigma_P = max(A_P / v_P, 0.0)
    # rho-INDEPENDENT athermal residual: above the kink-pair knee sigma_P would
    # collapse to 0, leaving the Taylor term (rho-dependent) as the ONLY
    # resistance -- which conflates the obstacle and the (exhaustible) carrier.
    # A finite floor keeps a resistance that survives exhaustion, so elasticity
    # carries stress above it while the mobile content mediates a bounded strain.
    sigma_P_floor = float(getattr(disl_cfg, 'peierls_floor_min_Pa', 0.0))
    return max(sigma_P, sigma_P_floor)


def calibrate_peierls_floor(disl_cfg, eta0, b, T_cal, eps_ref=None, S_kB=None):
    """Solve the additive (phi=1, rho-independent) Peierls-floor enthalpy so the
    floor equals ``peierls_floor_min_MPa`` at ``T_cal``, holding a PHYSICAL
    activation entropy.

    The phi=1 floor inverts  eps_ref = eta0 * exp(-(H_P - T*S_P - sigma*v_P)/kT)
    to  sigma_P = [ H_P - T*S_P - kT*ln(eta0/eps_ref) ] / v_P.  Requiring
    sigma_P(T_cal) = sigma_min gives the closed form

        H_P = sigma_min*v_P + k*T_cal*( S_kB + ln(eta0/eps_ref) ).

    The bracket shows the role of entropy: S_kB = -ln(eta0/eps_ref) (~ -37 kB)
    exactly cancels the temperature growth of the rate term (athermal floor);
    less-negative S (down to ~-10 kB) leaves a floor that decreases with T (the
    physical Peierls/DBTT collapse) while staying >= sigma_min over [T, T_cal].
    Returns a dict of the calibrated parameters and the floor at a few T.
    """
    v_P = max(float(getattr(disl_cfg, 'peierls_v0_b3', 5.0)), 1e-6) * b**3
    if eps_ref is None:
        eps_ref = float(getattr(disl_cfg, 'peierls_epsdot_ref', None)
                        or getattr(disl_cfg, 'flow_epsdot_ref', 1e-5))
    eps_ref = max(float(eps_ref), 1e-300)
    eta0 = max(float(eta0), 1e-300)
    ln_rate = np.log(eta0 / eps_ref)
    sig_min = max(float(getattr(disl_cfg, 'peierls_floor_min_MPa', 1.0)), 0.0) * 1e6

    S_athermal = -ln_rate  # in kB units
    if S_kB is None:
        S_kB = float(getattr(disl_cfg, 'peierls_S_kB', 0.0))
        if S_kB == 0.0:                      # not set by the user -> physical default
            S_kB = max(min(S_athermal + 7.0, -10.0), -50.0)  # ~ -30 kB for eta0/eps=1e16
    # Clamp to the physical mechanics range and warn outside it.  Never allow S
    # more negative than just above the athermal point: that inverts the floor's
    # T-trend and drives H_P < 0 (seen as a zero floor at all T below T_cal).
    S_floor = S_athermal + 0.5
    S_clamped = max(min(S_kB, -1.0), S_floor)
    S_was_clamped = bool(S_kB < S_floor)

    H_P_J = sig_min * v_P + KB * T_cal * (S_clamped + ln_rate)
    H_P_eV = H_P_J / EV_TO_J

    disl_cfg.use_peierls_floor = True
    disl_cfg.peierls_H0_eV = float(H_P_eV)
    disl_cfg.peierls_S_kB = float(S_clamped)

    info = {
        'H_P_eV': float(H_P_eV), 'S_kB': float(S_clamped), 'v_P_b3': float(getattr(disl_cfg, 'peierls_v0_b3', 5.0)),
        'S_athermal_kB': float(S_athermal), 'ln_rate': float(ln_rate),
        'sigma_min_MPa': sig_min / 1e6, 'T_cal_K': float(T_cal),
        'inverted_trend': bool(S_clamped < S_athermal),
        'unphysical_negative_H': bool(H_P_eV < 0.0),
        'S_was_clamped': S_was_clamped,
    }
    return info


def flow_stress_two_branch(rho, T, eps_ref, plast_model, disl_cfg, b,
                           return_branches=False):
    """Additive two-branch flow stress (model A):

        sigma_y(rho, T) = sigma_Peierls(T) + sigma_Taylor(rho, T)

    * sigma_Peierls : phi_P = 1, own (H_P, v_P, S_P); INDEPENDENT of rho.
                      Low-temperature thermal stress (kink-pair / lattice
                      friction).
    * sigma_Taylor  : phi_T = 1/(2 b sqrt(rho)), forest depinning.  The rho
                      dependence and its hardening->softening turnover
                      (paper Eq. 8 spinodal) live ENTIRELY in this branch.

    Both branches are evaluated at the SAME local equivalent plastic strain
    rate ``eps_ref`` so the result is a genuine sum of resistances along the
    glide path, not two independent rate problems.  This is the single source
    of truth for the yield surface; the diagnostic and the update both call it,
    which prevents the previous inline copies from drifting apart.

    Guardrail for (A): rho-hardening feeds ONLY sigma_Taylor.  sigma_Peierls
    ignores rho by construction, so the two contributions are not double
    counted, and plastic work is later accumulated once from the total
    sigma_y * d(eps_p), never per branch.
    """
    rho = np.asarray(rho, dtype=float)
    rho_safe = np.maximum(rho, 1e6)
    phi_max = float(getattr(disl_cfg, 'phi_plastic_max', 20.0))
    sigma_T = arrhenius_taylor_flow_stress(rho_safe, T, eps_ref, plast_model, b,
                                           phi_plastic_max=phi_max)
    # ATHERMAL TAYLOR FLOOR.  The Arrhenius inversion above is a purely
    # thermal-barrier picture: when A + kT*ln(eps_dot/prefactor) < 0 the
    # zero-stress thermal rate already exceeds the imposed rate and the
    # branch clamps to ZERO -- above the Peierls knee (~600 K for W at
    # quasi-static rates) the material is left with no strength at all,
    # which is the high-T plastic-work runaway (sigma_y(900K) = 0.1 MPa,
    # rho explodes, Wp/Wext -> 1e5%).  Physically, forest junctions have a
    # MECHANICAL threshold ~ alpha*G*b*sqrt(rho) that thermal activation
    # can assist but not erase at laboratory rates (large activation
    # volume); for W at 900 K forest hardening is essentially athermal.
    # The floor is a lower bound on the depinning stress; the Eq.-8
    # hardening->softening physics still acts wherever the Arrhenius
    # branch exceeds the floor.  Set taylor_athermal_alpha = 0 to recover
    # the unfloored behaviour.
    alpha_ath = float(getattr(disl_cfg, 'taylor_athermal_alpha', 0.2))
    if alpha_ath > 0.0:
        G_mod = None
        for _src_obj in (plast_model, getattr(plast_model, 'mat', None),
                         getattr(plast_model, 'material', None)):
            if _src_obj is None:
                continue
            G_mod = getattr(_src_obj, 'G', None)
            if G_mod is not None:
                break
        if G_mod is None:
            G_mod = 161e9  # tungsten shear modulus fallback
        sigma_T = np.maximum(sigma_T, alpha_ath * float(G_mod) * b * np.sqrt(rho_safe))
    sigma_P = peierls_flow_stress(T, eps_ref, plast_model, disl_cfg, b)
    sigma_y = sigma_P + sigma_T
    if return_branches:
        return sigma_y, sigma_P, sigma_T
    return sigma_y


def plastic_flow_diagnostics(
    rho_gp: np.ndarray,
    sigma_gp: np.ndarray,
    mat: ElasticProperties,
    T: float,
    plast_model: PlasticityModel,
    disl_cfg: DislocationConfig,
) -> dict:
    """Return scalar diagnostics for the current plastic flow threshold.

    This is intentionally side-effect free.  It recomputes the same rate-dependent
    flow stress used by ``update_plasticity`` so single-case runs can distinguish
    between: (i) no plasticity because the barrier is truly hard, (ii) no plasticity
    because the EXP_floor inversion is floor-limited, and (iii) global yielding.
    """
    out = {}
    try:
        b = mat.b
        G = mat.G
        rho_safe = np.maximum(np.asarray(rho_gp, dtype=float), 1e6)
        seq, _, _ = _von_mises_plane_strain(np.asarray(sigma_gp, dtype=float), mat.nu)
        delta = 1.0 / (2.0 * np.sqrt(rho_safe))
        phi_raw = delta / b
        phi = np.minimum(phi_raw, max(float(getattr(disl_cfg, 'phi_plastic_max', 20.0)), 1.0))

        # Two-branch additive flow stress (single source of truth): Peierls
        # (phi=1, rho-independent) + Taylor (phi=1/(2b sqrt(rho))), both at the
        # same reference rate.
        eps_ref = max(float(getattr(disl_cfg, 'flow_epsdot_ref', 1e-5)), 1e-30)
        sigma_y, sigma_P, sigma_T = flow_stress_two_branch(
            rho_safe, T, eps_ref, plast_model, disl_cfg, b, return_branches=True)
        denom = np.maximum(sigma_y, 1e-30)
        over = np.maximum(seq - sigma_y, 0.0)
        dgamma_uncapped = 0.999 * over / (3.0 * G)
        dep_cap = float(getattr(disl_cfg, 'max_plastic_strain_increment', np.inf))
        cap_gamma = np.sqrt(2.0/3.0) * dep_cap if np.isfinite(dep_cap) and dep_cap > 0 else np.inf

        out.update({
            'sigma_eq_mean': float(np.nanmean(seq)),
            'sigma_eq_max': float(np.nanmax(seq)),
            'sigma_y_min': float(np.nanmin(sigma_y)),
            'sigma_y_mean': float(np.nanmean(sigma_y)),
            'sigma_y_max': float(np.nanmax(sigma_y)),
            'sigma_T_min': float(np.nanmin(sigma_T)),
            'sigma_T_mean': float(np.nanmean(sigma_T)),
            'sigma_T_max': float(np.nanmax(sigma_T)),
            'sigma_Peierls': float(sigma_P),
            'sigma_eq_over_sigma_y_max': float(np.nanmax(seq / denom)),
            'yield_frac': float(np.mean(seq > sigma_y)),
            'flow_dgamma_uncapped_max': float(np.nanmax(dgamma_uncapped)),
            'flow_dgamma_cap': float(cap_gamma) if np.isfinite(cap_gamma) else 0.0,
            'flow_cap_frac': float(np.mean(dgamma_uncapped > cap_gamma)) if np.isfinite(cap_gamma) else 0.0,
            'flow_phi_mean': float(np.nanmean(phi)),
            'flow_phi_max': float(np.nanmax(phi)),
        })

        if getattr(plast_model, 'uses_embedded_stress_barrier', False):
            sigc = plast_model._exp_sigc(T)
            sigma_ref = max(float(getattr(plast_model.p, 'exp_sigma_deriv_min_frac', 1e-4)) * sigc, 1.0)
            v_ref = max(plast_model.v(np.array([sigma_ref]), T)[0], 1e-40)
            prefactor = plast_model.p.eta0 * b * v_ref / (delta**4)
            target = KB * T * np.log(np.maximum(prefactor / eps_ref, 1e-300))
            G0 = plast_model.G_barrier(np.zeros_like(rho_safe), T)
            Gfloor = plast_model._exp_floor_eV(T) * EV_TO_J
            solved = (target < G0) & (target > Gfloor)
            zero_stress = target >= G0
            floor_limited = target <= Gfloor
            out.update({
                'flow_Gtarget_eV_min': float(np.nanmin(target / EV_TO_J)),
                'flow_Gtarget_eV_mean': float(np.nanmean(target / EV_TO_J)),
                'flow_Gtarget_eV_max': float(np.nanmax(target / EV_TO_J)),
                'flow_DG0_eV': float(np.nanmean(G0 / EV_TO_J)),
                'flow_DGfloor_eV': float(Gfloor / EV_TO_J),
                'flow_vstar_ref_b3': float(v_ref / b**3),
                'flow_status_zero_stress_frac': float(np.mean(zero_stress)),
                'flow_status_solved_frac': float(np.mean(solved)),
                'flow_status_floor_limited_frac': float(np.mean(floor_limited)),
            })
        else:
            out.update({
                'flow_Gtarget_eV_min': 0.0,
                'flow_Gtarget_eV_mean': 0.0,
                'flow_Gtarget_eV_max': 0.0,
                'flow_DG0_eV': 0.0,
                'flow_DGfloor_eV': 0.0,
                'flow_vstar_ref_b3': 0.0,
                'flow_status_zero_stress_frac': 0.0,
                'flow_status_solved_frac': 0.0,
                'flow_status_floor_limited_frac': 0.0,
            })
    except Exception:
        # Diagnostics must not crash a mechanics run.
        keys = ['sigma_eq_mean','sigma_eq_max','sigma_y_min','sigma_y_mean','sigma_y_max',
                'sigma_T_min','sigma_T_mean','sigma_T_max','sigma_Peierls',
                'sigma_eq_over_sigma_y_max','yield_frac','flow_dgamma_uncapped_max',
                'flow_dgamma_cap','flow_cap_frac','flow_phi_mean','flow_phi_max',
                'flow_Gtarget_eV_min','flow_Gtarget_eV_mean','flow_Gtarget_eV_max',
                'flow_DG0_eV','flow_DGfloor_eV','flow_vstar_ref_b3',
                'flow_status_zero_stress_frac','flow_status_solved_frac',
                'flow_status_floor_limited_frac']
        out = {k: 0.0 for k in keys}
    return out


def build_elem_adjacency(mesh):
    """Element face-adjacency for conservative cell-centred transport.

    Returns (ei, ej, cond): for every interior edge shared by two triangles,
    the element indices and a geometric conductance = edge_length /
    centroid_distance.  No-flux on domain boundary (unshared edges dropped),
    so a diffusion built on this graph conserves sum(rho*area) exactly.
    Cache the result on the mesh; it depends only on geometry.
    """
    elems = mesh.elems
    ne = elems.shape[0]
    nodes = mesh.nodes
    centroids = nodes[elems].mean(axis=1)
    # each triangle has 3 edges (sorted node-pair keys)
    edge_map = {}
    for e in range(ne):
        n0, n1, n2 = elems[e]
        for a, b in ((n0, n1), (n1, n2), (n2, n0)):
            key = (a, b) if a < b else (b, a)
            edge_map.setdefault(key, []).append(e)
    ei, ej, cond = [], [], []
    for (a, b), els in edge_map.items():
        if len(els) != 2:
            continue                      # boundary edge -> no flux
        i, j = els
        edge_len = float(np.hypot(*(nodes[a] - nodes[b])))
        dctr = float(np.hypot(*(centroids[i] - centroids[j])))
        if dctr <= 0:
            continue
        ei.append(i); ej.append(j); cond.append(edge_len / dctr)
    return (np.asarray(ei, dtype=int),
            np.asarray(ej, dtype=int),
            np.asarray(cond, dtype=float))


def transport_rho_step(rho, adj, area_e, D_e, dt):
    """One conservative finite-volume diffusion step of the density field.

    Flux across face (i,j): F = D_face * cond_ij * (rho_i - rho_j), with
    D_face the face-average mobility-scaled diffusivity.  rho_i -= dt*F/area_i,
    rho_j += dt*F/area_j conserves total content (sum rho*area) with no-flux
    boundaries.  An explicit step is stable only if dt*D*cond/area is small;
    we sub-cycle to keep the per-face Courant number < 0.25.
    """
    ei, ej, cond = adj
    if ei.size == 0:
        return rho
    Dface = 0.5 * (D_e[ei] + D_e[ej])
    k = Dface * cond                                  # [m^2/s * 1] -> conductance
    amin = np.minimum(area_e[ei], area_e[ej])
    courant = dt * np.max(k / np.maximum(amin, 1e-30)) if k.size else 0.0
    nsub = int(min(64, max(1, np.ceil(courant / 0.25))))
    dts = dt / nsub
    out = rho.copy()
    for _ in range(nsub):
        F = k * (out[ei] - out[ej])
        np.add.at(out, ei, -dts * F / area_e[ei])
        np.add.at(out, ej,  dts * F / area_e[ej])
    return out


def _zero_plastic_info(ne: int) -> dict:
    z = np.zeros(ne, dtype=float)
    return {
        'dWp_requested_gp': z.copy(),
        'dWp_accepted_gp': z.copy(),
        'dep_eq_requested_gp': z.copy(),
        'dep_eq_accepted_gp': z.copy(),
        'thermo_scale_gp': np.ones(ne, dtype=float),
        'thermo_admissible_gp': z.copy(),
        'thermo_hazard_gp': z.copy(),
        'thermo_mode': 'off',
    }


def _kinetic_arrhenius_rate(
    rho_safe: np.ndarray,
    seq: np.ndarray,
    sigma_P: float,
    phi: np.ndarray,
    delta: np.ndarray,
    T: float,
    plast_model: PlasticityModel,
    disl_cfg: DislocationConfig,
    b: float,
) -> np.ndarray:
    """Kinetic Arrhenius equivalent plastic strain rate at the current stress.

    This is used by the thermodynamic Onsager/time-cone modes as the kinetic
    mobility/event clock.  The rate is still only a proposal; admissibility is
    enforced by the local free-energy relaxation distance in update_plasticity.
    """
    ne = len(seq)
    seq_drive = np.maximum(seq - sigma_P, 0.0)
    sigma_local = seq_drive * phi
    G_star = plast_model.G_barrier(sigma_local, T)

    # Prefactor: (b/delta)*(v*/delta^3)*eta0.
    # For EXP_floor, v*(sigma,T) is derived from -dG/dsigma; use the local
    # derivative at the local stress.  For rational_Hv, this returns the
    # fitted local activation volume.
    try:
        v_star = plast_model.v(np.maximum(sigma_local, 0.0), T)
    except Exception:
        v_star = plast_model.v(np.zeros(1), T)[0] * np.ones(ne)
    v_star = np.maximum(v_star, 1e-40)
    prefactor = plast_model.p.eta0 * b * v_star / np.maximum(delta, 1e-30)**4

    if T > 0:
        log_rate = np.log(np.maximum(prefactor, 1e-300)) - G_star / (KB * T)
        log_rate = np.clip(log_rate, -745, 80)
        rate = np.exp(log_rate)
    else:
        rate = np.zeros(ne)
    rate = np.minimum(rate, float(getattr(disl_cfg, 'dot_ep_max', np.inf)))
    rate = np.where(seq_drive > 0.0, rate, 0.0)
    return np.where(rate > 1e-30, rate, 0.0)


def update_plasticity(
    ep_gp: np.ndarray,           # (3, ne) plastic strain at GPs
    rho_gp: np.ndarray,          # (ne,) dislocation density
    sigma_gp: np.ndarray,        # (3, ne) stress at GPs
    mat: ElasticProperties,
    T: float,
    dt: float,
    plast_model: PlasticityModel,
    disl_cfg: DislocationConfig,
    return_info: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized plasticity update using the Arrhenius-Taylor model.

    In addition to the legacy explicit/flow-stress update, this routine now
    supports two thermodynamically admissible kinetic interpretations:

    * thermo_consistency_mode='onsager': continuous dissipative flow.  The
      Arrhenius rate is a mobility, the force is the local overstress, and the
      update is projected onto the local elastic relaxation distance.

    * thermo_consistency_mode='time_cone': hazard/time-cone flow.  The local
      Arrhenius clock advances only inside the admissible cone (positive
      thermodynamic force and non-negative local dissipation); the accepted
      increment is the event probability times the same relaxation distance.

    Both modes prevent the plastic branch from spending arbitrary energy by
    construction: the increment cannot overshoot the local decrease in elastic
    free energy available from the current overstress.  This is not a global
    Wp/Wext gate; it is a local thermodynamic admissibility projection.
    """
    ne = sigma_gp.shape[1]
    b = mat.b
    G = mat.G
    nu = mat.nu
    sqrt23 = np.sqrt(2.0 / 3.0)

    if not bool(getattr(disl_cfg, 'enable_plasticity', True)):
        z = np.zeros(ne)
        info = _zero_plastic_info(ne)
        return (ep_gp, np.clip(rho_gp, 1e6, disl_cfg.rho_cap), z, info) if return_info else (ep_gp, np.clip(rho_gp, 1e6, disl_cfg.rho_cap), z)

    # --- Deviatoric stress and local geometric amplification ---
    seq, norm_s, (sd_xx, sd_yy, txy) = _von_mises_plane_strain(sigma_gp, nu)
    rho_eval_floor = float(getattr(disl_cfg, 'mobile_rho_floor', 1e6))
    rho_safe = np.maximum(rho_gp, rho_eval_floor)
    delta = 1.0 / (2.0 * np.sqrt(rho_safe))
    phi_raw = delta / b
    phi_cap = max(float(getattr(disl_cfg, 'phi_plastic_max', np.inf)), 1.0)
    phi = np.minimum(phi_raw, phi_cap)

    # --- Production emission-derived Peierls--Taylor kinetics (v9.3) ---
    # The active production branch has no additive Peierls/Taylor yield stress.
    # It evaluates two scaled EXP-floor hazards derived from the active emission
    # surface, combines them as sequential bottlenecks, and converts the result
    # to plastic strain rate through a mobile-carrier Orowan relation.  Taylor
    # completion is a correlated multi-hit Poisson tail whose hit order grows
    # with forest density, preventing the independent-site rho^2 prefactor from
    # generating a nonphysical high-density strength downturn.
    kinetics_name = str(getattr(
        disl_cfg, 'bulk_kinetics_model', 'legacy_additive_flow_stress'
    )).lower()
    use_pt = kinetics_name in {
        'emission_derived_peierls_taylor_multihit',
        'emission_derived_pt', 'peierls_taylor_multihit', 'pt_multihit'
    }
    pt_diag = {}
    update_mode = str(getattr(disl_cfg, 'plastic_update_mode', 'explicit_rate')).lower()
    thermo_mode = str(getattr(disl_cfg, 'thermo_consistency_mode', 'off')).lower()

    if use_pt:
        from .emission_derived_plasticity import (
            EmissionDerivedPeierlsTaylorModel,
            config_from_dislocation_config,
        )
        pt_model = EmissionDerivedPeierlsTaylorModel(
            config_from_dislocation_config(disl_cfg)
        )
        pt_diag = pt_model.rates(seq, rho_safe, T, b)
        dot_ep_kin = np.asarray(
            pt_diag['equivalent_plastic_rate_s'], dtype=float
        )
        sigma_y = np.zeros_like(seq)
        sigma_P = 0.0
        sigma_T = np.zeros_like(seq)
        overstress = np.maximum(seq, 0.0)
        # The production model is a kinetic event law.  If an old input file
        # leaves thermodynamic consistency off, use the local time-cone update
        # rather than silently returning to a quasi-static yield surface.
        if thermo_mode == 'off':
            thermo_mode = 'time_cone'
        update_mode = 'explicit_rate'
    else:
        # Explicit legacy ablation: additive two-branch flow stress
        # sigma_y = sigma_P + sigma_T(rho).
        eps_ref = max(float(getattr(disl_cfg, 'flow_epsdot_ref', 1e-5)), 1e-30)
        sigma_y, sigma_P, sigma_T = flow_stress_two_branch(
            rho_safe, T, eps_ref, plast_model, disl_cfg, b,
            return_branches=True)
        dot_ep_kin = _kinetic_arrhenius_rate(
            rho_safe, seq, sigma_P, phi, delta, T, plast_model, disl_cfg, b)
        overstress = np.maximum(seq - sigma_y, 0.0)

    # Maximum local relaxation distance for a radial return.  This is the
    # thermodynamic cone: for 0 <= dgamma <= dgamma_eq the local elastic energy
    # decreases rather than the plastic branch creating energy.
    dgamma_eq = 0.999 * overstress / (3.0 * G)

    if thermo_mode == 'onsager':
        # Continuous dissipative flow: qdot = L(A,T) A.  The Arrhenius rate
        # gives the equivalent plastic strain rate proposal, projected onto the
        # admissible relaxation distance.
        dep_eq_prop = dot_ep_kin * max(dt, 0.0)
        dep_eq_max = dgamma_eq / sqrt23
        frac_max = float(getattr(disl_cfg, 'thermo_onsager_max_fraction', 1.0))
        if np.isfinite(frac_max) and frac_max > 0:
            dep_eq_max_step = frac_max * dep_eq_max
        else:
            dep_eq_max_step = dep_eq_max
        dep_eq = np.minimum(dep_eq_prop, dep_eq_max_step)
        dep_eq = np.minimum(dep_eq, dep_eq_max)
        dgamma = sqrt23 * np.maximum(dep_eq, 0.0)
        dep_eq_requested = dep_eq_prop
        hazard = np.zeros(ne)

    elif thermo_mode == 'time_cone':
        # Time-cone/hazard flow: the event clock only advances in the
        # admissible cone.  The expected accepted increment is the cone event
        # probability times the relaxation distance.
        dep_event = max(float(getattr(disl_cfg, 'thermo_event_strain', 1e-4)), 1e-16)
        H = dot_ep_kin * max(dt, 0.0) / dep_event
        H = np.clip(H, 0.0, 80.0)
        p_event = 1.0 - np.exp(-H)
        dep_eq_max = dgamma_eq / sqrt23
        dep_eq = p_event * dep_eq_max
        dgamma = sqrt23 * np.maximum(dep_eq, 0.0)
        dep_eq_requested = dot_ep_kin * max(dt, 0.0)
        hazard = H

    elif update_mode == 'flow_stress':
        # Legacy quasi-static return: go essentially all the way to the
        # rate-dependent flow surface within the load increment.
        dgamma = dgamma_eq.copy()
        dep_cap = float(getattr(disl_cfg, 'max_plastic_strain_increment', np.inf))
        if np.isfinite(dep_cap) and dep_cap > 0:
            dgamma = np.minimum(dgamma, sqrt23 * dep_cap)
        dep_eq_requested = dgamma / sqrt23
        hazard = np.zeros(ne)

    else:
        # Legacy explicit rate mode with a thermodynamic radial-return cap.
        dgamma = sqrt23 * dot_ep_kin * max(dt, 0.0)
        dgamma = np.minimum(dgamma, dgamma_eq)
        dep_cap = float(getattr(disl_cfg, 'max_plastic_strain_increment', np.inf))
        if np.isfinite(dep_cap) and dep_cap > 0:
            dgamma = np.minimum(dgamma, sqrt23 * dep_cap)
        dep_eq_requested = dot_ep_kin * max(dt, 0.0)
        hazard = np.zeros(ne)

    # Enforce the accepted-increment limiter for *all* plasticity paths.
    # Earlier thermodynamic/time-cone branches limited the hazard clock but not
    # the final relaxation distance when seq became very large.  That allowed
    # dep_eq_accepted ~ O(1) in one load step for W/Si failure cases.  The cap
    # below is a numerical substep admissibility limit: it prevents a single
    # accepted update from spending an unresolvable amount of plastic work.
    dep_eq_uncapped = dgamma / sqrt23
    dep_caps = []
    dep_cap_user = float(getattr(disl_cfg, 'max_plastic_strain_increment', np.inf))
    if np.isfinite(dep_cap_user) and dep_cap_user > 0:
        dep_caps.append(dep_cap_user)
    dep_cap_thermo = float(getattr(disl_cfg, 'thermo_max_dep_increment', np.inf))
    if bool(getattr(disl_cfg, 'thermo_adaptive_substepping', False)) and np.isfinite(dep_cap_thermo) and dep_cap_thermo > 0:
        dep_caps.append(dep_cap_thermo)
    dep_cap_final = min(dep_caps) if dep_caps else np.inf
    dep_was_limited = np.zeros_like(dep_eq_uncapped, dtype=float)
    if np.isfinite(dep_cap_final) and dep_cap_final > 0:
        dep_limited = np.minimum(dep_eq_uncapped, dep_cap_final)
        dep_was_limited = (dep_eq_uncapped > dep_cap_final * (1.0 + 1e-12)).astype(float)
        dgamma = sqrt23 * dep_limited
    else:
        dep_limited = dep_eq_uncapped

    # Accepted equivalent plastic strain rate.
    dep_eq_accepted = dgamma / sqrt23
    dot_ep_gp = np.where(dt > 0, dep_eq_accepted / dt, 0.0)

    # Flow direction (in-plane Voigt).
    safe_norm = np.maximum(norm_s, 1e-30)
    n_xx = sd_xx / safe_norm
    n_yy = sd_yy / safe_norm
    n_xy = txy / safe_norm

    # Plastic strain increment.
    ep_gp[0, :] += dgamma * 1.5 * n_xx
    ep_gp[1, :] += dgamma * 1.5 * n_yy
    ep_gp[2, :] += dgamma * 1.5 * n_xy

    # Thermodynamic plastic work diagnostics.  The requested work is what the
    # old pre-return stress-power estimate would have spent.  The accepted work
    # integrates the stress along the accepted local return path.  This is the
    # value the main code should accumulate as Wp.
    seq_after = np.maximum(seq - 3.0 * G * dgamma, 0.0)
    dWp_requested_gp = seq * np.maximum(dep_eq_requested, 0.0)
    if bool(getattr(disl_cfg, 'thermo_use_avg_stress_work', True)):
        dWp_accepted_gp = 0.5 * (seq + seq_after) * np.maximum(dep_eq_accepted, 0.0)
    else:
        dWp_accepted_gp = np.maximum(sigma_y, 0.0) * np.maximum(dep_eq_accepted, 0.0)
    scale = np.ones(ne)
    denom = np.maximum(dWp_requested_gp, 1e-300)
    scale = np.where(dWp_requested_gp > 0, dWp_accepted_gp / denom, 1.0)
    admissible = (overstress > 0).astype(float)

    # --- Dislocation evolution (eq 14 from paper) ---
    rho = rho_gp
    # Frank-Read-style bulk multiplication.  In the 'sources-only' picture this
    # is suppressed (bulk_mult_frac -> 0): new content is generated only at the
    # tip and transported in; the bulk keeps only sinks (recovery) + transport.
    mult_frac = float(getattr(disl_cfg, 'bulk_mult_frac', 1.0))
    drho_store = mult_frac * disl_cfg.k_store * np.sqrt(np.maximum(rho, 1e-30)) / b * dot_ep_gp
    drho_dyn = disl_cfg.k_dyn * rho * dot_ep_gp

    gamma_static = np.zeros(ne)
    if disl_cfg.use_static_recovery:
        Tm = mat.Tm
        if T / max(Tm, 1) >= disl_cfg.Tfrac_on:
            kev = 8.617333262e-5
            Dl = (disl_cfg.Dl0a * np.exp(-disl_cfg.Ea_eV / (kev * max(T, 1e-9))) +
                  disl_cfg.Dl0b * np.exp(-disl_cfg.Eb_eV / (kev * max(T, 1e-9))))
            x = disl_cfg.kpp * mat.E * b**4 * np.sqrt(np.maximum(rho, 1e-30)) / max(KB * T, 1e-30)
            x = np.clip(x, 0, 700)
            exm1 = np.maximum(np.exp(x) - 1, 1e-30)
            rho32 = rho * np.sqrt(np.maximum(rho, 1e-30))
            gamma_static = disl_cfg.kprime * (Dl / b) * rho32 / exm1
            gamma_static = np.clip(gamma_static, 0, disl_cfg.gamma_cap)

    if bool(getattr(disl_cfg, 'freeze_rho', False)):
        rho_trial = rho.copy()
    else:
        rho_trial = rho + dt * (drho_store - drho_dyn - gamma_static)

    rel_cap = float(getattr(disl_cfg, 'max_rho_relative_increment', np.inf))
    if np.isfinite(rel_cap) and rel_cap > 0:
        rho_upper_step = rho * (1.0 + rel_cap)
        rho_lower_step = rho / (1.0 + rel_cap)
        rho_trial = np.minimum(np.maximum(rho_trial, rho_lower_step), rho_upper_step)

    # finite-content exhaustion: the mobile density swept to sinks while
    # mediating this step's plastic shear is removed, d(rho) = -dgamma/(b*L).
    # Applied AFTER the relative-increment cap so the content budget stays
    # conservative (the cap throttles storage/recovery, not the sink flux).
    if not bool(getattr(disl_cfg, 'freeze_rho', False)) and \
            bool(getattr(disl_cfg, 'exhaustion_enabled', False)):
        L_sink = max(float(getattr(disl_cfg, 'glide_to_sink_m', 1e-5)), 1e-12)
        rho_trial = rho_trial - dgamma / (b * L_sink)

    rho_gp = np.clip(rho_trial, rho_eval_floor, disl_cfg.rho_cap)

    info = {
        'dWp_requested_gp': dWp_requested_gp,
        'dWp_accepted_gp': dWp_accepted_gp,
        'dep_eq_requested_gp': dep_eq_requested,
        'dep_eq_accepted_gp': dep_eq_accepted,
        'dep_eq_uncapped_gp': dep_eq_uncapped,
        'dep_eq_limited_gp': dep_was_limited,
        'dep_eq_cap': np.full(ne, dep_cap_final if np.isfinite(dep_cap_final) else np.nan),
        'thermo_scale_gp': scale,
        'thermo_admissible_gp': admissible,
        'thermo_hazard_gp': hazard,
        'thermo_mode': thermo_mode,
        'bulk_kinetics_model': kinetics_name,
        'bulk_pt_active': bool(use_pt),
        'pt_peierls_rate_gp': np.asarray(pt_diag.get('peierls_rate_s', np.zeros(ne)), dtype=float),
        'pt_taylor_single_hit_rate_gp': np.asarray(pt_diag.get('taylor_single_hit_rate_s', np.zeros(ne)), dtype=float),
        'pt_taylor_completion_rate_gp': np.asarray(pt_diag.get('taylor_completion_rate_s', np.zeros(ne)), dtype=float),
        'pt_series_rate_gp': np.asarray(pt_diag.get('series_rate_s', np.zeros(ne)), dtype=float),
        'pt_taylor_m_eff_gp': np.asarray(pt_diag.get('taylor_m_eff', np.ones(ne)), dtype=float),
        'pt_rho_mobile_gp': np.asarray(pt_diag.get('rho_mobile_m2', np.zeros(ne)), dtype=float),
        'pt_G_peierls_eV_gp': np.asarray(pt_diag.get('G_peierls_eV', np.zeros(ne)), dtype=float),
        'pt_G_taylor_eV_gp': np.asarray(pt_diag.get('G_taylor_eV', np.zeros(ne)), dtype=float),
    }
    return (ep_gp, rho_gp, dot_ep_gp, info) if return_info else (ep_gp, rho_gp, dot_ep_gp)


def arrhenius_taylor_flow_stress(
    rho: np.ndarray, T: float, eps_dot: float,
    plast_model: PlasticityModel, b: float,
    phi_plastic_max: float = 20.0,
) -> np.ndarray:
    """
    Compute macroscopic Arrhenius-Taylor flow stress.

    For rational_Hv, preserve the older analytical inversion.  For EXP_floor,
    numerically solve DeltaG(phi*sigma,T)=kBT ln(prefactor/epsdot), because
    the stress dependence is already embedded in DeltaG.
    """
    rho = np.asarray(rho, dtype=float)
    rho_safe = np.maximum(rho, 1e6)

    if getattr(plast_model, 'uses_embedded_stress_barrier', False):
        return arrhenius_taylor_flow_stress_numeric(rho_safe, T, eps_dot, plast_model, b, phi_plastic_max)

    v_star = plast_model.v(np.zeros(1), T)[0]
    v_star = max(v_star, 1e-36)
    H0 = plast_model.H(np.zeros(1))[0]
    S0 = plast_model.S(np.zeros(1), T)[0]
    # Invert eps_dot = eta0 * (b*v*/delta^4) * exp(-(H - T*S - sigma*phi*v*)/kT)
    # with delta = 1/(2 sqrt(rho))  =>  b*v*/delta^4 = 16 * b * v* * rho^2.
    # Solving for the local depinning stress and dividing by phi = 1/(2 b sqrt(rho)):
    #   sigma = (2 b sqrt(rho)/v*) * [ (H - T S) + kT ln( eps_dot / (eta0 * 16 b v* rho^2) ) ].
    # eta0 belongs INSIDE the log with the rate prefactor (not as KB*ln(eta0), which is
    # dimensionally J/K and sign-wrong); this is the same grouping the Peierls branch and
    # _kinetic_arrhenius_rate use, so the threshold and the kinetic rate are now consistent.
    A = H0 - T * S0
    prefactor = plast_model.p.eta0 * 16.0 * b * v_star * rho_safe**2
    log_arg = np.maximum(eps_dot / np.maximum(prefactor, 1e-300), 1e-300)
    bracket = A + KB * T * np.log(log_arg)
    sigma_flow = (2 * b * np.sqrt(rho_safe) / v_star) * bracket
    return np.maximum(sigma_flow, 0)


def arrhenius_taylor_flow_stress_numeric(
    rho: np.ndarray, T: float, eps_dot: float,
    plast_model: PlasticityModel, b: float,
    phi_plastic_max: float = 20.0,
) -> np.ndarray:
    """Numerically invert a full stress-biased barrier for Taylor flow."""
    rho = np.asarray(rho, dtype=float)
    rho_safe = np.maximum(rho, 1e6)
    delta = 1.0 / (2.0 * np.sqrt(rho_safe))
    phi = np.minimum(delta / b, max(float(phi_plastic_max), 1.0))

    sigma_ref = max(float(getattr(plast_model.p, 'exp_sigma_deriv_min_frac', 1e-4)) * plast_model._exp_sigc(T), 1.0)
    v_star = max(plast_model.v(np.array([sigma_ref]), T)[0], 1e-40)
    prefactor = plast_model.p.eta0 * b * v_star / (delta**4)
    target = KB * T * np.log(np.maximum(prefactor / max(eps_dot, 1e-300), 1e-300))

    # --- Correlated multi-hit Taylor renewal (opt-in) ---------------------
    # The independent-site inversion above has an effective attempt rate that
    # grows like rho^2 (the b*v*/delta^4 prefactor), which inverts to an
    # UNPHYSICAL softening of the flow stress at high density.  Physically, at
    # small forest spacing the junctions inside one elastic/topological
    # correlation length are not independent strain sources: a correlated
    # segment must complete m(rho) cooperative depinning hits within one
    # renewal time t_c before it glides.  Replacing N_site*h1 with
    # N_corr*(1/t_c)*P_>=m(n_c h1 t_c) suppresses the effective rate at high rho
    # and removes the inversion, while the m=1 / low-rho limit reproduces the
    # independent result EXACTLY (so this is a safe extension).
    #
    # We apply the correction as a transform of `target` (the required single-
    # junction barrier): with P_target = t_c*eta0*n_c*exp(-target/kT),
    #   Lambda* = Poisson-tail^{-1}_m(P_target),  G* = -kT ln(Lambda*/(n_c t_c eta0)).
    # For m->1 and small P_target, G* -> target identically.
    dl = getattr(plast_model, '_disl_cfg', None)
    if dl is not None and bool(getattr(dl, 'taylor_multihit', False)) and _gammaincinv is not None:
        rho_c = max(float(getattr(dl, 'taylor_corr_rho_c', 1e14)), 1e6)
        t_c = max(float(getattr(dl, 'taylor_renewal_time_s', 1e-9)), 1e-30)
        m_max = max(float(getattr(dl, 'taylor_m_max', 5.0)), 1.0)
        p_m = max(float(getattr(dl, 'taylor_m_exponent', 1.0)), 1e-6)
        eta0 = max(float(plast_model.p.eta0), 1e-300)
        n_c = np.maximum(np.sqrt(rho_safe / rho_c), 1.0)            # constraints per correlated segment
        r = (rho_safe / rho_c) ** p_m
        m_eff = 1.0 + (m_max - 1.0) * r / (1.0 + r)                 # smooth hit number m(rho) in [1, m_max]
        kT = KB * T
        # target completion probability per renewal window (clamped to a valid CDF range)
        P_tar = np.clip(t_c * eta0 * n_c * np.exp(-np.minimum(target / kT, 700.0)), 1e-300, 1.0 - 1e-12)
        Lam = _gammaincinv(m_eff, P_tar)                            # inverse Poisson upper tail
        Lam = np.maximum(Lam, 1e-300)
        Gstar = -kT * np.log(np.maximum(Lam / (n_c * t_c * eta0), 1e-300))
        # G(sigma) is DECREASING in sigma, so a LOWER target barrier => higher
        # flow stress.  The cooperative requirement (m>1) makes completion rarer,
        # which lowers G* below the independent target => stiffening.  Take the
        # min so the correction can only stiffen, never soften (m=1 => G*==target).
        target = np.minimum(Gstar, target)

    G0 = plast_model.G_barrier(np.zeros_like(rho_safe), T)
    need = target < G0
    sigma_local_flow = np.zeros_like(rho_safe)
    if not np.any(need):
        return sigma_local_flow

    sigc = plast_model._exp_sigc(T)
    lo = np.zeros_like(rho_safe[need])
    hi = np.full_like(lo, max(10.0 * sigc, 1e10))
    target_need = target[need]

    for _ in range(12):
        Ghi = plast_model.G_barrier(hi, T)
        bad = Ghi > target_need
        if not np.any(bad):
            break
        hi[bad] *= 2.0

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        Gmid = plast_model.G_barrier(mid, T)
        above = Gmid > target_need
        lo = np.where(above, mid, lo)
        hi = np.where(above, hi, mid)

    sigma_local_flow[need] = hi
    return np.maximum(sigma_local_flow / np.maximum(phi, 1e-30), 0.0)
